import time
import sys

def construct():
    start_time = time.time()
    deadline = 150.0  # 150s limit, keep margin
    
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

    def solve(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to sets for faster lookup, but keep original for iteration
        domains = [set(l) for l in lists]
        
        # Precompute degrees for tie-breaking
        degrees = [len(adj[i]) for i in range(n)]
        
        # Heuristic: Order vertices by degree (descending) to start with most constrained?
        # Actually, MRV is dynamic. We need a data structure or just scan.
        # For n=400, scanning is fine.
        
        color = [-1] * n
        
        # Forward checking: maintain domains
        # We need to be able to undo.
        
        def is_valid(v, c):
            for u in adj[v]:
                if color[u] == c:
                    return False
            return True

        def backtrack(v_count):
            if time.time() > deadline:
                return False
            if v_count == n:
                return True
            
            # Find unassigned vertex with smallest domain (MRV)
            # We can just iterate through all unassigned vertices
            # Optimization: Keep a list of unassigned vertices
            # Simpler: Just find the one with min domain size among unassigned
            min_size = float('inf')
            best_v = -1
            for i in range(n):
                if color[i] == -1:
                    # Calculate current domain size
                    # Domain is lists[i] minus colors of neighbors
                    # Since we are backtracking, we can compute this on the fly
                    # Or maintain domains explicitly.
                    # Given n is small, on-the-fly is safer to avoid state bugs.
                    
                    # Check if any neighbor is colored
                    # If a neighbor has color c, c is not available
                    # available = lists[i] - {color[u] for u in adj[i] if color[u] != -1}
                    
                    # To speed up, we can just count available colors
                    taken = set()
                    for u in adj[i]:
                        if color[u] != -1:
                            taken.add(color[u])
                    
                    # available = [c for c in lists[i] if c not in taken]
                    # size = len(available)
                    
                    # If size is 0, return False (dead end)
                    # If size < min_size, update best_v
                    # If size == 1, we can break early (forced move)
                    
                    avail_count = 0
                    for c in lists[i]:
                        if c not in taken:
                            avail_count += 1
                            if avail_count == 1 and min_size > 1:
                                # Potential early exit if we find a 1
                                pass 
                            if avail_count >= min_size:
                                break
                    # Re-calc properly if we broke early? No, just compute count.
                    # Let's just compute the list to be sure for the next step
                    # It's faster to just compute the list once.
                    
                    # Let's refine:
                    # available = [c for c in lists[i] if c not in taken]
                    # size = len(available)
                    
                    # We need the actual list for assignment.
                    # Let's store the best_v and its available list.
                    
                    # To avoid double work, let's just pick the vertex with min available.
                    # We'll do a simple scan.
                    pass
            
            # The above scan is O(n * deg). n=400, deg~4-6. 400*6 = 2400 ops. 
            # Depth 400. 400 * 2400 = 960,000 ops per level? No, per node in search tree.
            # Backtracking tree could be large if bad choices.
            # MRV helps a lot.
            
            # Let's implement the scan properly.
            best_v = -1
            best_avail = None
            min_avail_size = float('inf')
            
            for i in range(n):
                if color[i] == -1:
                    taken = set()
                    for u in adj[i]:
                        if color[u] != -1:
                            taken.add(color[u])
                    
                    avail = [c for c in lists[i] if c not in taken]
                    size = len(avail)
                    if size == 0:
                        return False
                    if size < min_avail_size:
                        min_avail_size = size
                        best_v = i
                        best_avail = avail
                        if size == 1:
                            break
                    elif size == min_avail_size:
                        # Tie break by degree
                        if degrees[i] > degrees[best_v]:
                            best_v = i
                            best_avail = avail
            
            if best_v == -1:
                return True # All assigned? Should have caught v_count==n
            
            # Try colors in best_avail
            # Order colors by least constraining value (LCV)?
            # For now, just iterate.
            for c in best_avail:
                color[best_v] = c
                if backtrack(v_count + 1):
                    return True
                color[best_v] = -1
            
            return False

        if backtrack(0):
            # Ensure types are int
            return [int(c) for c in color]
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5:
            # Safety: if running out of time, return what we have or empty?
            # Task says: "An instance whose colouring breaks a list or an edge scores 0... 
            # but a wrong length or type makes the whole result invalid."
            # If we return None, it's invalid.
            # If we return a partial/invalid coloring, it scores 0 for that instance.
            # We must return a list of 4 lists.
            # If we haven't solved it, we can't return a valid coloring.
            # Given the small size, we should solve it easily.
            pass
        
        n, edges, lists = instance(splits, seed)
        sol = solve(n, edges, lists)
        if sol is None:
            # Fallback: return a dummy list of 0s (will score 0 for this instance)
            # But we must maintain type and length.
            sol = [0] * n
        out.append(sol)
        
    return out

# To ensure it runs immediately when imported/executed as expected
# The prompt asks to "Rewrite the program... Output the COMPLETE program".
# The entry point is construct().

if __name__ == '__main__':
    # Just for local testing
    res = construct()
    print(f"Length: {len(res)}")
    for i, r in enumerate(res):
        print(f"Instance {i}: len={len(r)}, first few={r[:5]}")
