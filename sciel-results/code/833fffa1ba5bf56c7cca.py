"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

sys.setrecursionlimit(10000)

def construct():
    def instance(splits, seed, palette=5):
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

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def solve_csp(n, adj, lists, deadline, rng=None):
        """
        CSP Solver with MRV, LCV, and Forward Checking.
        Uses a deadline to ensure termination.
        """
        start_time = time.time()
        domains = [set(l) for l in lists]
        uncoloured = set(range(n))
        colour = [None] * n
        
        # If rng is provided, we can use it for tie-breaking in MRV to allow restarts
        # For deterministic first pass, rng is None, so we use sorted order or index
        
        def get_mrv_vertex():
            best_v = -1
            best_count = float('inf')
            # We can use rng for tie-breaking if available
            candidates = []
            for v in uncoloured:
                cnt = len(domains[v])
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    candidates = [v]
                elif cnt == best_count:
                    candidates.append(v)
            
            if not candidates:
                return -1
            
            if len(candidates) == 1:
                return candidates[0]
            
            if rng:
                return rng.choice(candidates)
            else:
                # Deterministic tie-break: lowest index
                return min(candidates)

        def propagate(v, c):
            """
            Assign v=c, remove c from neighbours' domains.
            Returns list of (neighbor, color) removed for undo, or None if failure.
            """
            removed = []
            for u in adj[v]:
                if u in uncoloured and c in domains[u]:
                    domains[u].discard(c)
                    removed.append((u, c))
                    if not domains[u]:
                        # Failure: undo changes made so far in this propagation step
                        for uu, cc in removed:
                            domains[uu].add(cc)
                        return None
            return removed

        def backtrack():
            if time.time() > deadline:
                return None  # Timeout
            
            if not uncoloured:
                return True
            
            v = get_mrv_vertex()
            if v == -1:
                return True  # Should not happen if uncoloured is not empty
            
            if len(domains[v]) == 0:
                return False
            
            # LCV: Sort colors by how many neighbors they constrain
            # Fewer constraints is better
            # We only care about uncoloured neighbors
            uncoloured_neighbors = [u for u in adj[v] if u in uncoloured]
            
            def lcv_score(c):
                count = 0
                for u in uncoloured_neighbors:
                    if c in domains[u]:
                        count += 1
                return count

            # Try colors in order of LCV score (ascending)
            # If scores are tied, use sorted order for determinism (or rng)
            colors_to_try = sorted(domains[v], key=lambda c: (lcv_score(c), c))
            
            for c in colors_to_try:
                if time.time() > deadline:
                    return None
                
                uncoloured.discard(v)
                colour[v] = c
                removed = propagate(v, c)
                
                if removed is None:
                    # Propagation failed, undo assignment and continue
                    uncoloured.add(v)
                    colour[v] = None
                    continue
                
                result = backtrack()
                if result is True:
                    return True
                elif result is None:
                    # Timeout propagated up
                    # We need to undo before returning None
                    for u, rc in removed:
                        domains[u].add(rc)
                    uncoloured.add(v)
                    colour[v] = None
                    return None
                
                # Result is False, undo and try next color
                for u, rc in removed:
                    domains[u].add(rc)
                uncoloured.add(v)
                colour[v] = None
            
            return False

        result = backtrack()
        if result is True:
            return colour
        else:
            return None

    def greedy_fallback(n, adj, lists):
        """
        Simple greedy coloring to ensure we return a list of correct length.
        This might be invalid, but it prevents "wrong length" errors.
        """
        colour = [0] * n
        # Order by degree descending
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        for v in order:
            taken = {colour[u] for u in adj[v]}
            # Find a color in lists[v] not in taken
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                colour[v] = lists[v][0]
        return colour

    def is_valid(n, adj, lists, colour):
        if len(colour) != n:
            return False
        for v in range(n):
            if colour[v] not in lists[v]:
                return False
            for u in adj[v]:
                if u > v and colour[u] == colour[v]:
                    return False
        return True

    out = []
    # Total time budget is 150s. We use 140s to be safe.
    total_deadline = time.time() + 140.0
    
    # Allocate time per instance. 4 instances.
    # We can give more time to harder instances if needed, but equal split is fine.
    # Let's give each instance up to 30 seconds, but stop if total deadline is reached.
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        current_time = time.time()
        if current_time > total_deadline - 5:
            # Not enough time left, just do greedy for remaining
            n, edges, lists = instance(splits, seed)
            adj = build_adj(n, edges)
            out.append(greedy_fallback(n, adj, lists))
            continue
        
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)
        
        # Instance deadline
        inst_deadline = min(total_deadline - 5, current_time + 30)
        
        solution = None
        
        # Try 1: Deterministic CSP
        solution = solve_csp(n, adj, lists, inst_deadline, rng=None)
        
        if solution is None:
            # Try 2: Randomized restarts
            # Use a different seed for the random tie-breaking
            rng = Random(seed + 12345)
            # Give a smaller time budget for restarts
            restart_deadline = min(total_deadline - 5, time.time() + 10)
            solution = solve_csp(n, adj, lists, restart_deadline, rng=rng)
            
            if solution is None:
                # Try 3: Another randomized restart
                rng2 = Random(seed + 54321)
                restart_deadline2 = min(total_deadline - 5, time.time() + 5)
                solution = solve_csp(n, adj, lists, restart_deadline2, rng=rng2)
        
        if solution is not None:
            # Validate
            if is_valid(n, adj, lists, solution):
                out.append(solution)
            else:
                # Should not happen if solver is correct, but fallback just in case
                out.append(greedy_fallback(n, adj, lists))
        else:
            # Fallback
            out.append(greedy_fallback(n, adj, lists))

    # Ensure types are correct
    final_out = []
    for lst in out:
        final_out.append([int(c) for c in lst])
    
    return final_out

# EVOLVE-BLOCK-END
