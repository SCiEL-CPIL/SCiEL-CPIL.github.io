import time
import random
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n = 16
    # All possible pairs
    all_pairs = list(combinations(range(n), 2))
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    
    # All possible triples
    all_triples = list(combinations(range(n), 3))
    
    # Map each triple to the indices of the 3 pairs it covers
    triple_to_pairs = {}
    for t in all_triples:
        a, b, c = t
        p1 = (a, b)
        p2 = (a, c)
        p3 = (b, c)
        triple_to_pairs[t] = (pair_to_idx[p1], pair_to_idx[p2], pair_to_idx[p3])

    # Precompute independence number for a given set of triple masks
    # To make this fast, we use a bitmask representation.
    # A set S is independent if no triple is a subset of S.
    # We want to minimize the maximum size of such S.
    
    def calc_alpha(triples):
        if not triples:
            return n
        masks = [ (1<<a)|(1<<b)|(1<<c) for a,b,c in triples ]
        
        # We want to find the max size k such that there exists a subset of size k 
        # that contains no triple.
        # Equivalently, we want to find the max k such that there is a mask m 
        # with popcount(m)=k and (m & tm) != tm for all tm in masks.
        
        # Optimization: 
        # If we have a "covering" set of points, alpha is small.
        # Let's just iterate all masks? 2^16 = 65536.
        # For each mask, check if it's independent.
        # This is O(2^n * T). T is ~20. 65536 * 20 = 1.3M ops.
        # In Python, this is ~0.1s per call. Too slow for many calls.
        
        # Better: 
        # We can compute the independence number of the hypergraph defined by 'triples'.
        # Let's use a recursive branching on points?
        # Or, since n=16 is small, we can precompute the "forbidden" masks?
        
        # Actually, let's use a smarter check.
        # A mask m is NOT independent if it contains any triple mask.
        # We can precompute a boolean array `is_forbidden` of size 2^n?
        # No, the triples change.
        
        # Let's try a heuristic first: 
        # Random sampling to estimate alpha? No, we need exact for pruning.
        
        # Let's optimize the loop.
        # Instead of iterating all masks, we can branch on points.
        # But for now, let's use the simple loop but with numpy if possible? 
        # No, standard lib.
        
        # Let's try to prune the mask iteration.
        # We want max popcount.
        # Iterate masks in decreasing order of popcount?
        # Generating combinations of size k is better.
        
        # Max possible alpha is n.
        # Min possible alpha is 3 (if we have a Steiner triple system covering all points? 
        # Actually, if we have a STS(16) which doesn't exist, but close).
        # For n=16, lower bound on alpha is roughly n / d_avg?
        
        # Let's just do the loop but break early if we find a large one?
        # No, we want the MAX. So we must check all.
        
        # Wait, if we find an independent set of size k, we don't stop. 
        # We want the largest.
        
        # Let's use a different approach:
        # The complement of an independent set is a vertex cover.
        # Minimizing alpha is equivalent to maximizing the vertex cover size? No.
        # Alpha is the size of the largest independent set.
        
        # Let's stick to the mask loop but optimize it.
        # We can represent the set of triple masks.
        # For a given mask m, is it independent?
        # Check if any tm is a subset of m.
        
        # We can precompute for each triple mask tm, the list of masks that contain it? No.
        
        # Let's just run the loop. If it's too slow, we'll reduce the number of evaluations.
        
        max_alpha = 0
        # We can iterate masks from 0 to 2^n - 1
        # To speed up, we can skip masks that are subsets of known independent sets?
        
        # Actually, there is a known algorithm for alpha of hypergraphs.
        # But for n=16, brute force is okay if we are careful.
        
        # Let's try to use bit operations.
        # For each mask m, we want to know if there exists tm in masks such that (m & tm) == tm.
        
        # We can precompute a "hit" table?
        # Let's just do the simple loop.
        
        for m in range(1 << n):
            indep = True
            for tm in masks:
                if (m & tm) == tm:
                    indep = False
                    break
            if indep:
                size = bin(m).count('1')
                if size > max_alpha:
                    max_alpha = size
                    # If max_alpha == n, we can break? No, n is the max possible.
                    # If we find n, it means the empty set of triples? No, m is full set.
                    # If m is full set, it contains all triples. So it's not independent unless no triples.
                    # So max_alpha < n if triples exist.
        return max_alpha

    # DLX Implementation for Exact Cover
    # We want to select a set of triples such that every pair is covered at most once.
    # This is not an exact cover, it's a partial cover.
    # We can model it as: Select triples such that no pair is repeated.
    # This is a matching in a 3-uniform hypergraph? No.
    # It's a packing.
    
    # We can use DLX by adding "dummy" columns for pairs that are NOT covered?
    # Or simply, we want to find a maximal packing.
    # DLX finds solutions to exact cover.
    # We can transform:
    # Columns: 120 pairs.
    # Rows: 120 triples + 120 "skip" rows?
    # If we include a "skip" row for a pair, it means the pair is not covered.
    # Then an exact cover corresponds to a valid packing where every pair is either covered by a triple or explicitly skipped.
    # This ensures linearity.
    
    # Number of columns: 120.
    # Number of rows: 120 (triples) + 120 (skips) = 240.
    
    # Build DLX matrix
    # Col j corresponds to pair j.
    # Row i (0..119) corresponds to triple i. It has 1s in columns corresponding to its 3 pairs.
    # Row i (120..239) corresponds to skip of pair i-120. It has 1 in column i-120.
    
    num_cols = len(all_pairs)
    num_rows = len(all_triples) + num_cols
    
    # DLX data structures
    # We use arrays for L, R, U, D, col, row
    L = [0] * (num_rows * 3 + num_cols + 1) # Max nodes: each triple has 3, each skip has 1. Total 3*120+120 = 480 nodes. Plus header.
    R = [0] * (num_rows * 3 + num_cols + 1)
    U = [0] * (num_rows * 3 + num_cols + 1)
    D = [0] * (num_rows * 3 + num_cols + 1)
    col_arr = [0] * (num_rows * 3 + num_cols + 1)
    row_arr = [0] * (num_rows * 3 + num_cols + 1)
    size_arr = [0] * (num_cols + 1) # Size of column j (number of 1s)
    
    node_count = 0
    # Initialize header row (node 0)
    for j in range(num_cols + 1):
        L[j] = j - 1
        R[j] = j + 1
    R[0] = num_cols
    L[num_cols] = 0
    
    # Initialize columns
    for j in range(1, num_cols + 1):
        U[j] = j
        D[j] = j
        size_arr[j] = 0
        
    # Add rows
    # Triples
    for i, t in enumerate(all_triples):
        pairs = triple_to_pairs[t]
        first_node = -1
        for p_idx in pairs:
            node_count += 1
            col_arr[node_count] = p_idx + 1 # 1-based col index
            row_arr[node_count] = i
            U[node_count] = U[col_arr[node_count]]
            D[node_count] = col_arr[node_count]
            D[U[node_count]] = node_count
            U[col_arr[node_count]] = node_count
            size_arr[col_arr[node_count]] += 1
            
            if first_node == -1:
                first_node = node_count
                L[node_count] = node_count
                R[node_count] = node_count
            else:
                # Insert at end of row
                L[node_count] = L[first_node]
                R[node_count] = first_node
                R[L[first_node]] = node_count
                L[first_node] = node_count
                
    # Skips
    for j in range(num_cols):
        node_count += 1
        col_arr[node_count] = j + 1
        row_arr[node_count] = len(all_triples) + j
        U[node_count] = U[col_arr[node_count]]
        D[node_count] = col_arr[node_count]
        D[U[node_count]] = node_count
        U[col_arr[node_count]] = node_count
        size_arr[col_arr[node_count]] += 1
        
        L[node_count] = node_count
        R[node_count] = node_count

    # DLX Search
    solution = []
    best_alpha = n
    best_system = []
    nodes_visited = 0
    deadline_reached = False
    
    def cover(c):
        R[L[c]] = R[c]
        L[R[c]] = L[c]
        i = D[c]
        while i != c:
            j = R[i]
            while j != i:
                U[D[j]] = U[j]
                D[U[j]] = D[j]
                size_arr[col_arr[j]] -= 1
                j = R[j]
            i = D[i]
            
    def uncover(c):
        i = U[c]
        while i != c:
            j = L[i]
            while j != i:
                size_arr[col_arr[j]] += 1
                D[U[j]] = j
                U[D[j]] = j
                j = L[j]
            i = U[i]
        R[L[c]] = c
        L[R[c]] = c

    def search():
        nonlocal best_alpha, best_system, nodes_visited, deadline_reached
        if deadline_reached:
            return
            
        nodes_visited += 1
        if nodes_visited % 1000 == 0:
            if time.time() > deadline:
                deadline_reached = True
                return
                
        if R[0] == 0:
            # Solution found
            # Extract triples
            sys_triples = []
            for r in solution:
                if r < len(all_triples):
                    sys_triples.append(all_triples[r])
            
            # Calculate alpha
            alpha = calc_alpha(sys_triples)
            if alpha < best_alpha:
                best_alpha = alpha
                best_system = sys_triples[:]
                if best_alpha <= 5: # Theoretical lower bound?
                    deadline_reached = True
            return

        # Choose column with minimum size
        c = R[0]
        min_size = size_arr[c]
        j = R[c]
        while j != 0:
            if size_arr[j] < min_size:
                min_size = size_arr[j]
                c = j
            j = R[j]
            
        if min_size == 0:
            return # Dead end

        cover(c)
        r = D[c]
        while r != c:
            if deadline_reached:
                break
            solution.append(row_arr[r])
            j = R[r]
            while j != r:
                cover(col_arr[j])
                j = R[j]
            search()
            j = L[r]
            while j != r:
                uncover(col_arr[j])
                j = L[j]
            solution.pop()
            r = D[r]
        uncover(c)

    # We can't search the whole space, it's too large.
    # We need to prune or limit.
    # The space of packings is huge.
    # However, DLX with "skips" ensures we enumerate all valid packings.
    # We will likely time out.
    # We need a heuristic to guide the search or limit depth.
    # Or, we can just run it and hope it finds a good one quickly?
    # Unlikely.
    
    # Alternative: Use the DLX structure but with a randomization or limited depth?
    # Or, use a different initialization.
    
    # Given the time limit, a full DLX search is infeasible.
    # Let's fall back to a faster heuristic if DLX is too slow, 
    # or modify the search to be stochastic.
    
    # Actually, let's just run the greedy packing first to get a baseline, 
    # then use the time remaining for a local search (SA) as before, 
    # but with the EXACT alpha calculation.
    # The "different strategy" is the use of EXACT alpha calculation 
    # to guide the local search, avoiding the heuristic errors of the previous SA.
    
    # 1. Greedy Initialization
    random.seed(42)
    used_pairs = set()
    current_system = []
    shuffled_triples = all_triples[:]
    random.shuffle(shuffled_triples)
    
    for t in shuffled_triples:
        a, b, c = t
        p1 = (a,b)
        p2 = (a,c)
        p3 = (b,c)
        if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
            current_system.append(t)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
    current_alpha = calc_alpha(current_system)
    best_alpha = current_alpha
    best_system = current_system[:]
    
    # 2. Local Search with Exact Alpha
    # Moves: Swap a triple for another valid one.
    # We maintain used_pairs set.
    
    def is_valid(t, used):
        a,b,c = t
        return (a,b) not in used and (a,c) not in used and (b,c) not in used
        
    def add_to_used(t, used):
        a,b,c = t
        used.add((a,b))
        used.add((a,c))
        used.add((b,c))
        
    def remove_from_used(t, used):
        a,b,c = t
        used.discard((a,b))
        used.discard((a,c))
        used.discard((b,c))

    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick a random triple in current system
        if not current_system:
            break
        idx = random.randint(0, len(current_system)-1)
        old_t = current_system[idx]
        
        # Try to find a replacement
        # Sample from all triples
        new_t = None
        for _ in range(20):
            cand = random.choice(all_triples)
            # Temporarily remove old_t to check validity of cand
            remove_from_used(old_t, used_pairs)
            if is_valid(cand, used_pairs):
                new_t = cand
                break
            add_to_used(old_t, used_pairs)
            
        if new_t:
            # Apply move
            remove_from_used(old_t, used_pairs)
            add_to_used(new_t, used_pairs)
            current_system[idx] = new_t
            
            new_alpha = calc_alpha(current_system)
            delta = new_alpha - current_alpha
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_alpha = new_alpha
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_system = current_system[:]
                    if best_alpha <= 5:
                        break
            else:
                # Revert
                remove_from_used(new_t, used_pairs)
                add_to_used(old_t, used_pairs)
                current_system[idx] = old_t
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_system

if __name__ == "__main__":
    res = construct()
    print(res)
