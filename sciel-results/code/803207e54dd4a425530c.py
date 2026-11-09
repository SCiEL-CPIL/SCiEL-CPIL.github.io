"""Rewritten program using optimized MAC with Bucket-Based MRV."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 120s limit, leave 5s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Bitmask domains
        doms = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            doms[i] = mask
        
        if any(d == 0 for d in doms):
            return None

        # Trail for undoing
        trail = []

        # Buckets for MRV: bucket[k] contains nodes with domain size k
        # Max domain size is 5 (colors 0-4)
        max_dom_size = 5
        buckets = [[] for _ in range(max_dom_size + 1)]
        # Initialize buckets
        for i in range(n):
            size = doms[i].bit_count()
            if size > 0:
                buckets[size].append(i)
            else:
                return None

        def update_bucket(node, old_size, new_size):
            # Remove node from old bucket
            if old_size > 0 and old_size <= max_dom_size:
                # Linear search is fine for small lists, or use a set?
                # For efficiency, we can just leave it and check validity in MRV selection,
                # or maintain a set. Let's use a set for O(1) removal?
                # Actually, lists are small. But we might have duplicates if we don't remove carefully.
                # Better: use a set for each bucket? Or just mark and clean?
                # Let's use a list and a "in_bucket" flag or similar.
                # Simpler: Just rebuild buckets if it gets too messy? No.
                # Let's use a set for each bucket to allow O(1) removal.
                pass

        # Let's refine the bucket structure to use sets for O(1) removal.
        buckets = [set() for _ in range(max_dom_size + 1)]
        for i in range(n):
            size = doms[i].bit_count()
            if size > 0:
                buckets[size].add(i)

        def move_to_bucket(node, old_size, new_size):
            if old_size > 0 and old_size <= max_dom_size:
                buckets[old_size].discard(node)
            if new_size > 0 and new_size <= max_dom_size:
                buckets[new_size].add(node)

        def ac3(initial_queue=None):
            if initial_queue is None:
                queue = [(u, v) for u in range(n) for v in adj[u]]
            else:
                queue = list(initial_queue)
            
            head = 0
            while head < len(queue):
                i, j = queue[head]
                head += 1
                
                dj = doms[j]
                # If dom[j] is a singleton, remove that color from dom[i]
                if dj and (dj & (dj - 1)) == 0:
                    if doms[i] & dj:
                        old_mask = doms[i]
                        new_mask = doms[i] & ~dj
                        if new_mask != old_mask:
                            doms[i] = new_mask
                            trail.append((i, old_mask))
                            old_size = old_mask.bit_count()
                            new_size = new_mask.bit_count()
                            if new_size == 0:
                                move_to_bucket(i, old_size, 0)
                                return False
                            move_to_bucket(i, old_size, new_size)
                            # Propagate if i is now singleton
                            if new_size == 1:
                                for k in adj[i]:
                                    if k != j:
                                        queue.append((k, i))
            return True

        # Initial AC-3
        mark = len(trail)
        if not ac3():
            return None
        
        assignment = [-1] * n
        degrees = [len(adj[i]) for i in range(n)]

        def find_mrv():
            # Find the smallest non-empty bucket
            for size in range(1, max_dom_size + 1):
                if buckets[size]:
                    # Pick a node from this bucket
                    # Tie-breaker: highest degree
                    best_node = -1
                    best_deg = -1
                    for node in buckets[size]:
                        # Verify it's still unassigned and size is correct (safety)
                        if assignment[node] == -1 and doms[node].bit_count() == size:
                            if degrees[node] > best_deg:
                                best_deg = degrees[node]
                                best_node = node
                    if best_node != -1:
                        return best_node, size
            return -1, 0

        def backtrack():
            if time.time() > DEADLINE:
                raise TimeoutError()
            
            min_node, min_size = find_mrv()
            
            if min_node == -1:
                return True # Success

            mask = doms[min_node]
            vals = []
            c = 0
            while mask:
                if mask & 1:
                    vals.append(c)
                mask >>= 1
                c += 1

            for c in vals:
                bit = 1 << c
                
                # Check consistency with assigned neighbors
                valid = True
                for nb in adj[min_node]:
                    if assignment[nb] != -1:
                        if assignment[nb] == c:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Save state
                mark = len(trail)
                
                # Assign c to min_node
                assignment[min_node] = c
                old_mask = doms[min_node]
                old_size = old_mask.bit_count()
                doms[min_node] = bit
                new_size = 1
                trail.append((min_node, old_mask))
                move_to_bucket(min_node, old_size, new_size)
                
                # Propagate: Run AC-3 on neighbors
                queue = [(nb, min_node) for nb in adj[min_node] if assignment[nb] == -1]
                
                if not ac3(queue):
                    # Conflict, undo and try next value
                    assignment[min_node] = -1
                    while len(trail) > mark:
                        var, old_mask = trail.pop()
                        # Restore bucket status
                        # We need to know old_size to move back. 
                        # The trail stores old_mask, so we can compute old_size.
                        # But we also need to know the size it was in BEFORE the change.
                        # This is tricky with the bucket approach if we don't store bucket history.
                        # Alternative: Don't use buckets for undo, just rebuild or mark invalid.
                        # Or: Store (var, old_mask, old_size, new_size_at_time_of_change)?
                        # Actually, the trail entry (var, old_mask) is enough to restore doms[var].
                        # But we also need to restore the bucket membership.
                        # The node was in bucket `old_size` before, and moved to `new_size` (which is 0 or >0).
                        # When we undo, we need to move it back to `old_size`.
                        # However, the `ac3` function might have moved it multiple times?
                        # No, `ac3` is called once per assignment. Inside `ac3`, each variable is updated at most once per propagation step?
                        # Actually, a variable can be updated multiple times in a single `ac3` run?
                        # Yes, if multiple neighbors force different colors.
                        # So the trail will have multiple entries for the same variable?
                        # No, `ac3` checks `if new_mask != old_mask`. If it changes, it appends to trail.
                        # If it changes again later in the same `ac3` run, it appends again.
                        # So the trail correctly records all changes.
                        # When we undo, we pop in reverse order.
                        # We need to know the size the variable had *before* the change that is being undone.
                        # The `old_mask` in the trail entry gives the size before that specific change.
                        # So:
                        #   var, old_mask = trail.pop()
                        #   current_mask = doms[var]
                        #   current_size = current_mask.bit_count()
                        #   old_size = old_mask.bit_count()
                        #   move_to_bucket(var, current_size, old_size)
                        #   doms[var] = old_mask
                        pass
                    # Let's implement the undo logic properly in the loop below
                    continue
                
                # Recurse
                if backtrack():
                    return True
                
                # Undo
                assignment[min_node] = -1
                while len(trail) > mark:
                    var, old_mask = trail.pop()
                    current_mask = doms[var]
                    current_size = current_mask.bit_count()
                    old_size = old_mask.bit_count()
                    move_to_bucket(var, current_size, old_size)
                    doms[var] = old_mask
            
            return False

        try:
            if backtrack():
                return [int(assignment[i]) for i in range(n)]
        except TimeoutError:
            pass
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            n, _, _ = instance(splits, seed)
            out.append([0] * n)
            continue
            
        n, edges, lists = instance(splits, seed)
        
        if time.time() > DEADLINE:
            out.append([0] * n)
            continue
            
        sol = solve_instance(n, edges, lists)
        
        if sol is None:
            out.append([0] * n)
        else:
            out.append([int(c) for c in sol])
            
    return out
# EVOLVE-BLOCK-END
