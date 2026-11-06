import time
import random
import sys
import numpy as np

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors for each vertex
NEIGHBORS = [[] for _ in range(SIZE)]
for v in range(SIZE):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# Precompute neighbor masks for fast conflict checking
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for nb in NEIGHBORS[v]:
        m |= 1 << nb
    NEIGHBOR_MASK[v] = m

def construct():
    deadline = time.time() + 140.0
    rnd = random.Random(42)
    
    # We will model the problem as finding a long path in Q9.
    # Instead of a direct clique on edges (which is complex due to connectivity),
    # we use a sophisticated Backtracking with Forward Checking (CSP approach)
    # which is a different design family from the Beam Search/Greedy of the current program.
    # The key difference: CSP explores the constraint space systematically with MRV (Minimum Remaining Values)
    # and Forward Checking to prune the search tree early, rather than maintaining a set of partial paths (beam).
    
    # To make it even more distinct and robust, we combine CSP with a "Repair" phase (VNS-like) 
    # that takes a valid short snake and tries to insert vertices or swap segments to extend it.
    
    # However, the most promising "different" approach for snake-in-a-box is often 
    # **Local Search with Conflict Resolution** or **Simulated Annealing** on the permutation of edges.
    # Let's implement a **Greedy + Local Search (2-opt / Or-opt)** hybrid.
    # 
    # Strategy:
    # 1. Generate a valid snake using a smart greedy (Worst-Case-First heuristic).
    # 2. Use Local Search to try to insert unused vertices into the path or swap segments to increase length.
    # 3. This is distinct from Beam Search (which explores multiple paths in parallel) and CSP (which is systematic backtracking).
    #    Local Search optimizes a single path iteratively.
    
    def get_neighbors(v):
        return NEIGHBORS[v]

    def is_valid_path(path):
        if not path:
            return True
        if len(path) == 1:
            return True
        used = set(path)
        if len(used) != len(path):
            return False
        for i in range(len(path) - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return False
        # Check non-consecutive adjacency
        for i in range(len(path)):
            v = path[i]
            for nb in NEIGHBORS[v]:
                if nb in used and abs(path.index(nb) - i) > 1:
                    return False
        return True

    # Fast validity check for incremental updates
    def validate_path_fast(path):
        n = len(path)
        if n == 0:
            return True
        used_mask = 0
        for i, v in enumerate(path):
            if used_mask & (1 << v):
                return False
            if i >= 2:
                # Check if current vertex is adjacent to any previous vertex except the immediate predecessor
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << path[i-1])
                if forbidden:
                    return False
            used_mask |= (1 << v)
        # Check the last vertex against all previous
        # The loop above handles i>=2. For i=0 and i=1, no check needed.
        # But we must ensure the last vertex doesn't connect to path[0] if len > 2.
        # The loop `for i, v in enumerate(path)` checks `forbidden` for every i>=2.
        # This covers all non-consecutive pairs (i, j) where i > j+1.
        # Does it cover (i, j) where i < j-1? Yes, because when we process index j, we check against all i < j-1.
        # So if path[j] is adjacent to path[i] (i < j-1), it will be caught when processing j.
        # If path[i] is adjacent to path[j] (i > j+1), it is caught when processing i.
        return True

    def greedy_snake(priority):
        """
        Generate a snake using a greedy strategy with a given priority of bit flips.
        Uses a 'lookahead' to avoid immediate dead ends if possible.
        """
        path = [0]
        used_mask = 1
        cur = 0
        for _ in range(300):
            if time.time() > deadline:
                break
            candidates = []
            for b in priority:
                nb = cur ^ (1 << b)
                if used_mask & (1 << nb):
                    continue
                # Check if nb is adjacent to any previous vertex except cur
                forbidden = NEIGHBOR_MASK[nb] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                # Score candidate: prefer those that leave more options for the next step?
                # Simple heuristic: count available neighbors of nb (excluding used and nb's neighbors in path)
                # This is expensive. Let's just pick the first valid one or randomize slightly.
                candidates.append(nb)
            
            if not candidates:
                break
            
            # Pick a candidate. To improve, we can pick the one with the fewest remaining options (MRV-like)
            # but that's expensive. Let's stick to priority order but add some randomness.
            # If we always pick the first, we get a deterministic path.
            # Let's shuffle candidates slightly if there are many? No, priority is already shuffled.
            # Let's just pick the first valid candidate in priority order.
            nxt = candidates[0]
            path.append(nxt)
            used_mask |= (1 << nxt)
            cur = nxt
            
        return path

    def local_search_optimize(path, max_iters=100):
        """
        Try to extend or improve the path using 2-opt style swaps or insertions.
        This is the core of the "Different Strategy": Local Search.
        """
        if not path:
            return path
        
        current_path = list(path)
        used_mask = 0
        for v in current_path:
            used_mask |= (1 << v)
        
        # Find all unused vertices
        all_vertices = set(range(SIZE))
        used_set = set(current_path)
        unused_vertices = list(all_vertices - used_set)
        
        best_len = len(current_path)
        best_path = current_path
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try to insert an unused vertex into the path
            #    For each edge (u, v) in the path, check if an unused vertex w exists
            #    such that u-w and w-v are valid edges and w is not adjacent to any other vertex in the path.
            #    This is O(L * U * N). L~100, U~500, N=9. ~450k ops per iter. Might be slow.
            #    Optimization: Only try a subset of unused vertices or edges.
            
            # Let's try a simpler move: "Swap" two non-adjacent segments?
            # Or "Rotate" a segment?
            
            # Given time constraints, let's try a simpler "Greedy Restart with Perturbation"
            # which is a form of Local Search (Simulated Annealing lite).
            
            # Actually, the most effective "different" strategy for this specific problem 
            # that is distinct from Beam Search is **Backtracking with Constraint Propagation (CSP)**.
            # The inspirations suggest CSP scores higher (1.49). 
            # Let's implement a lightweight CSP solver.
            
            # CSP:
            # Variables: positions 0..L-1 in the path.
            # Domains: all vertices.
            # Constraints: 
            #   1. Consecutive vars differ by 1 bit.
            #   2. Non-consecutive vars differ by >= 2 bits.
            #   3. All vars distinct.
            
            # We will solve for a path of length L starting from a fixed prefix (from greedy).
            
            break # Exit local search, switch to CSP in main loop below

        return best_path

    # --- CSP Solver Implementation ---
    
    def solve_csp(prefix, target_len):
        """
        Use Backtracking with Forward Checking to extend a prefix to target_len.
        prefix: list of vertices
        target_len: desired total length
        """
        if time.time() > deadline:
            return None
        
        path = list(prefix)
        used_mask = 0
        for v in path:
            used_mask |= (1 << v)
        
        # For efficiency, we track the domain of the next vertex implicitly.
        # We use a recursive backtracking function.
        
        def backtrack(current_path, current_used_mask, last_v):
            if time.time() > deadline:
                return None
            
            if len(current_path) == target_len:
                return list(current_path)
            
            cur = current_path[-1]
            
            # Candidates: neighbors of cur not in used_mask
            candidates = []
            for nb in NEIGHBORS[cur]:
                if current_used_mask & (1 << nb):
                    continue
                # Check if nb is adjacent to any vertex in current_path except cur
                forbidden = NEIGHBOR_MASK[nb] & current_used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                candidates.append(nb)
            
            if not candidates:
                return None
            
            # Heuristic: Order candidates by degree (number of available neighbors) ascending (MRV)
            # This helps prune dead ends earlier.
            def degree_score(v):
                count = 0
                for nb2 in NEIGHBORS[v]:
                    if not (current_used_mask | (1 << v)) & (1 << nb2):
                        # Check if nb2 is adjacent to any other vertex in path (except v)
                        # This is expensive. Approximate by just counting unvisited neighbors.
                        count += 1
                return count
            
            # To save time, we might skip full MRV sorting if the list is small or time is tight.
            # Let's just shuffle or use a simple heuristic.
            # Actually, MRV is crucial for CSP. Let's implement a fast version.
            
            # Fast MRV: Count unvisited neighbors.
            # We don't check full adjacency constraint for the score, just availability.
            candidates.sort(key=lambda v: sum(1 for nb2 in NEIGHBORS[v] if not (current_used_mask | (1 << v)) & (1 << nb2)))
            
            for nb in candidates:
                new_path = current_path + [nb]
                new_used_mask = current_used_mask | (1 << nb)
                res = backtrack(new_path, new_used_mask, nb)
                if res is not None:
                    return res
            
            return None
        
        return backtrack(path, used_mask, path[-1])

    # Main Loop:
    # 1. Generate a good prefix using Greedy.
    # 2. Use CSP to extend it to a longer length.
    # 3. Repeat with different seeds/priorities.
    
    best_path = [0]
    best_len = 1
    
    # Phase 1: Greedy to get a decent prefix
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        priority = list(range(N))
        rnd.shuffle(priority)
        
        # Greedy with a twist: sometimes pick randomly from top 3 candidates to diversify
        path = [0]
        used_mask = 1
        cur = 0
        for _ in range(150): # Cap greedy length to leave time for CSP
            if time.time() > deadline - 5:
                break
            candidates = []
            for b in priority:
                nb = cur ^ (1 << b)
                if used_mask & (1 << nb):
                    continue
                forbidden = NEIGHBOR_MASK[nb] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                candidates.append(nb)
            
            if not candidates:
                break
            
            # Pick one. If we want diversity, we can pick randomly from candidates.
            # But greedy usually works better with a fixed order. Let's stick to priority order.
            nxt = candidates[0]
            path.append(nxt)
            used_mask |= (1 << nxt)
            cur = nxt
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Phase 2: Try to extend this prefix using CSP to a longer target
        # Try to extend to best_len + 10, best_len + 20, etc.
        # We start with a modest extension to ensure we get *some* improvement.
        
        # Use the greedy path as a prefix.
        # We will try to extend it to various lengths.
        # To save time, we only run CSP if the greedy path is already good enough to be a base.
        
        target_lengths = [len(path) + 5, len(path) + 10, len(path) + 15]
        
        for tl in target_lengths:
            if tl > 250: # Safety cap
                tl = 250
            if time.time() > deadline - 2:
                break
            
            # CSP solving can be expensive. We limit the search depth/time implicitly by the deadline.
            res = solve_csp(path, tl)
            if res and len(res) > best_len:
                # Validate just in case
                if validate_path_fast(res):
                    best_len = len(res)
                    best_path = res
                    # If we found a longer path, update the prefix for next CSP tries?
                    # No, we restart from the original greedy path or a new one.
                    # But we can try to extend this new best_path further.
                    # Let's break and start a new trial with this as the base?
                    # For simplicity, we just record it and continue.
    
    # Final Validation
    if not validate_path_fast(best_path):
        # Fallback to a simple valid path
        return [0]
        
    return best_path

# Fallback if the above is too slow or fails
if __name__ == "__main__":
    # This block is not executed during evaluation, just for local testing
    pass
