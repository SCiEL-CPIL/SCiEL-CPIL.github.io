# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square modulo 205.
    
    Strategy:
    1. Compute forbidden differences (nonzero quadratic residues mod 205).
    2. Build conflict bitmask: conflict[i] has bit j set if i and j cannot both be in set.
    3. Use branch-and-bound maximum independent set with upper bound pruning.
    4. Combine with local search for polishing.
    5. Return the best solution found within time budget.
    """
    m = 205
    deadline = time.time() + 620  # 10s safety margin before 630s kill
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Precompute: for each i, conflict[i] is a bitmask of all j that conflict with i
    conflict = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i] |= (1 << j)
    
    def can_add(cand, selected_mask):
        """Check if cand can be added to the set represented by selected_mask."""
        return (conflict[cand] & selected_mask) == 0
    
    def greedy_random(seed):
        """Random greedy construction using bitmask conflict checks."""
        rng = random.Random(seed)
        candidates = list(range(m))
        rng.shuffle(candidates)
        selected_mask = 0
        selected_list = []
        for cand in candidates:
            if can_add(cand, selected_mask):
                selected_mask |= (1 << cand)
                selected_list.append(cand)
        return selected_mask, selected_list
    
    def local_search(selected_mask, selected_list, rng, max_iters=2000):
        """1-add and 1-swap local search using bitmask conflict checks."""
        best_mask = selected_mask
        best_list = list(selected_list)
        best_size = len(best_list)
        
        no_improve = 0
        iters = 0
        while iters < max_iters and no_improve < 150:
            iters += 1
            improved = False
            
            # Try 1-add: find a candidate that can be added
            candidates_to_add = [c for c in range(m) if not (best_mask >> c) & 1]
            rng.shuffle(candidates_to_add)
            for cand in candidates_to_add:
                if can_add(cand, best_mask):
                    best_mask |= (1 << cand)
                    best_list.append(cand)
                    best_size += 1
                    improved = True
                    no_improve = 0
                    break
                else:
                    no_improve += 1
            
            if improved:
                continue
            
            # Try 1-swap: remove one element, add another
            swap_order = list(range(len(best_list)))
            rng.shuffle(swap_order)
            for idx in swap_order:
                rem = best_list[idx]
                temp_mask = best_mask & ~(1 << rem)
                
                # Find a candidate not in temp_mask that can be added
                candidates_to_add = [c for c in range(m) if not (temp_mask >> c) & 1]
                rng.shuffle(candidates_to_add)
                for cand in candidates_to_add:
                    if can_add(cand, temp_mask):
                        # Perform the swap
                        best_list[idx] = cand
                        best_mask = temp_mask | (1 << cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
        
        return best_mask, best_list, best_size
    
    def upper_bound(candidates_mask, current_size):
        """
        Compute an upper bound on the size of an independent set from candidates_mask.
        Uses a greedy maximal matching approximation: repeatedly pick a vertex with
        highest degree in the candidate subgraph, count it, and remove it and its neighbors.
        The remaining vertices can all be added (optimistic bound).
        """
        if candidates_mask == 0:
            return current_size
        
        # Count bits in mask
        def popcount(x):
            c = 0
            while x:
                x &= x - 1
                c += 1
            return c
        
        # Greedy bound: repeatedly select a vertex, remove it and its neighbors
        # The number of selections is a lower bound on the independence number,
        # but we want an upper bound. Use: |candidates| - minimum vertex cover approx.
        # Simpler upper bound: use the fact that in any independent set, 
        # no two adjacent vertices can both be selected.
        # Upper bound via greedy coloring of the conflict graph restricted to candidates.
        
        # For a quick upper bound: the size of a maximum matching gives a lower bound
        # on vertex cover, and |V| - |max matching| is an upper bound on independence number.
        # But computing max matching is expensive.
        
        # Simpler: use the bound that in any graph, alpha(G) <= |V| - |max matching|
        # And |max matching| >= |V| / (2 * max_degree + 1) ... this is getting complex.
        
        # Practical upper bound: greedy set cover approximation
        # The independence number is at most the size of a minimum vertex cover + 1... 
        # Actually, let's use a simpler heuristic upper bound:
        # Sort candidates by degree (in the candidate subgraph), and greedily
        # estimate how many can be selected.
        
        # For speed, use a simple bound: the number of candidates minus 
        # a fraction based on average degree.
        n_cands = popcount(candidates_mask)
        if n_cands == 0:
            return current_size
        
        # Compute average degree in the candidate subgraph
        total_deg = 0
        for i in range(m):
            if (candidates_mask >> i) & 1:
                # Count neighbors of i that are also in candidates_mask
                deg = popcount(conflict[i] & candidates_mask)
                total_deg += deg
        
        avg_deg = total_deg / n_cands if n_cands > 0 else 0
        
        # Upper bound: in a graph with average degree d, the independence number
        # is at most |V| / (1 + d/2) ... this isn't tight.
        # Use a simpler bound: alpha <= |V| * (1 + ln(d+1)) / (d+1) for d-regular graphs
        # For safety, use: alpha <= |V| - floor(|E| / |V|) where |E| = total_deg / 2
        n_edges = total_deg // 2
        bound = n_cands - n_edges // n_cands if n_cands > 0 else 0
        return current_size + max(bound, 0)
    
    def branch_and_bound(candidates_mask, current_size, current_list, best_size_ref, deadline, max_nodes=1000000):
        """
        Branch and bound for maximum independent set.
        candidates_mask: bitmask of vertices still available
        current_size: number of vertices in current independent set
        current_list: list of vertices in current independent set
        best_size_ref: [int] reference to best size found
        Returns: (best_mask, best_list, best_size)
        """
        best_mask = 0
        best_list = []
        best_size = 0
        nodes_visited = 0
        
        # Use a stack-based DFS to avoid recursion depth issues
        # State: (candidates_mask, current_mask, current_list_tuple)
        # For efficiency, store current_mask and derive list when needed
        
        stack = [(candidates_mask, 0, tuple())]
        
        while stack and time.time() < deadline:
            if nodes_visited > max_nodes:
                break
            
            cands, cur_mask, cur_list = stack.pop()
            nodes_visited += 1
            
            # Compute upper bound
            ub = upper_bound(cands, len(cur_list))
            if ub <= best_size:
                continue
            
            # If no candidates left, check if this is a valid solution
            if cands == 0:
                if len(cur_list) > best_size:
                    best_size = len(cur_list)
                    best_mask = cur_mask
                    best_list = list(cur_list)
                continue
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with highest degree in the candidate subgraph
            best_v = -1
            best_deg = -1
            temp_cands = cands
            while temp_cands:
                i = (temp_cands & (-temp_cands)).bit_length() - 1
                temp_cands &= temp_cands - 1
                deg = (conflict[i] & cands).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = i
            
            if best_v == -1:
                continue
            
            # Branch 1: include best_v
            new_cands = cands & ~(1 << best_v)
            # Remove all neighbors of best_v from candidates
            new_cands &= ~conflict[best_v]
            new_mask = cur_mask | (1 << best_v)
            new_list = cur_list + (best_v,)
            
            # Check if including best_v gives a better upper bound
            ub_include = upper_bound(new_cands, len(new_list))
            if ub_include > best_size:
                stack.append((new_cands, new_mask, new_list))
            
            # Branch 2: exclude best_v
            new_cands_exclude = cands & ~(1 << best_v)
            ub_exclude = upper_bound(new_cands_exclude, len(cur_list))
            if ub_exclude > best_size:
                stack.append((new_cands_exclude, cur_mask, cur_list))
        
        return best_mask, best_list, best_size
    
    # Main search loop
    best_mask = 0
    best_list = []
    best_size = 0
    
    # Start with deterministic greedy (in order)
    sel = []
    sel_mask = 0
    for cand in range(m):
        if can_add(cand, sel_mask):
            sel.append(cand)
            sel_mask |= (1 << cand)
    best_list = sel
    best_size = len(sel)
    best_mask = sel_mask
    
    # Try branch-and-bound first with a time limit
    bnb_deadline = time.time() + 300  # Use 300s for BnB
    full_cands = (1 << m) - 1
    bnb_mask, bnb_list, bnb_size = branch_and_bound(
        full_cands, 0, tuple(), [best_size], bnb_deadline, max_nodes=500000
    )
    
    if bnb_size > best_size:
        best_mask = bnb_mask
        best_list = bnb_list
        best_size = bnb_size
    
    # Polish with local search
    if time.time() < deadline:
        res_mask, res_list, res_size = local_search(best_mask, best_list, random.Random(42))
        if res_size > best_size:
            best_mask = res_mask
            best_list = res_list
            best_size = res_size
    
    # Try multiple random restarts with local search
    seed = 1
    while time.time() < deadline:
        # Random greedy start
        start_mask, start_list = greedy_random(seed)
        
        # Local search
        res_mask, res_list, res_size = local_search(start_mask, start_list, random.Random(seed))
        
        if res_size > best_size:
            best_mask = res_mask
            best_list = res_list
            best_size = res_size
        
        seed += 1
        
        # Also try local search on the current best every few iterations
        if seed % 3 == 0:
            res_mask, res_list, res_size = local_search(best_mask, best_list, random.Random(seed + 10000))
            if res_size > best_size:
                best_mask = res_mask
                best_list = res_list
                best_size = res_size
    
    return np.array(best_list, dtype=np.int64)

# EVOLVE-BLOCK-END
