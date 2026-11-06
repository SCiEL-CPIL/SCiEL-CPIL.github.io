import time
import random
from itertools import combinations

def _precompute_k222s(n):
    """
    Precompute all K(2,2,2) configurations.
    A K(2,2,2) is defined by 3 disjoint pairs: {a,a'}, {b,b'}, {c,c'}.
    The 8 transversal triples are all combinations choosing one from each pair.
    Returns:
    - all_triples: list of all C(n,3) triples as tuples
    - triple_to_idx: dict mapping triple to index
    - triple_to_k222s: list of lists, mapping triple index to list of K(2,2,2) indices it belongs to
    - k222_list: list of frozensets of triple indices for each K(2,2,2)
    """
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    triple_to_k222s = [[] for _ in range(num_triples)]
    k222_list = []
    
    # Generate all K(2,2,2) configurations
    # Choose 6 points out of n, then partition into 3 pairs
    for points in combinations(range(n), 6):
        # Generate all perfect matchings of these 6 points
        # Use a recursive generator
        def gen_matchings(pts):
            if not pts:
                yield []
                return
            first = pts[0]
            rest = pts[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for m in gen_matchings(new_rest):
                    yield [(first, second)] + m
        
        for matching in gen_matchings(points):
            # Matching is a list of 3 tuples: (a,a'), (b,b'), (c,c')
            # The 8 triples are formed by choosing one from each pair
            k_triples = []
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        t = tuple(sorted((matching[0][x], matching[1][y], matching[2][z])))
                        k_triples.append(t)
            
            k_indices = frozenset(triple_to_idx[t] for t in k_triples)
            k222_idx = len(k222_list)
            k222_list.append(k_indices)
            
            for idx in k_indices:
                triple_to_k222s[idx].append(k222_idx)
                
    return all_triples, triple_to_idx, triple_to_k222s, k222_list

def _greedy_construct(all_triples, triple_to_k222s, k222_list, order, time_limit=5.0):
    """
    Greedy construction using a specific order of triples.
    Uses a counter array to track how many triples are present in each K(2,2,2).
    """
    num_triples = len(all_triples)
    num_k222s = len(k222_list)
    
    # Counter for each K(2,2,2): how many of its 8 triples are currently present
    k_counts = [0] * num_k222s
    present = [False] * num_triples
    
    # Process triples in the given order
    for idx in order:
        # Check if adding this triple would complete any K(2,2,2)
        can_add = True
        for k_idx in triple_to_k222s[idx]:
            if k_counts[k_idx] == 7:  # 7 already present, adding makes 8
                can_add = False
                break
        
        if can_add:
            present[idx] = True
            # Update counts for all K(2,2,2)s containing this triple
            for k_idx in triple_to_k222s[idx]:
                k_counts[k_idx] += 1
    
    return [i for i in range(num_triples) if present[i]]

def _local_search(initial_set, all_triples, triple_to_k222s, k222_list, time_limit=10.0):
    """
    Local search using swaps to improve the set.
    Uses a counter array for efficient conflict checking.
    """
    num_triples = len(all_triples)
    num_k222s = len(k222_list)
    
    # Build initial state
    present = [False] * num_triples
    for idx in initial_set:
        present[idx] = True
    
    k_counts = [0] * num_k222s
    for idx in initial_set:
        for k_idx in triple_to_k222s[idx]:
            k_counts[k_idx] += 1
    
    current_size = len(initial_set)
    best_size = current_size
    best_set = set(initial_set)
    
    start_time = time.time()
    current_set = set(initial_set)
    
    while time.time() - start_time < time_limit:
        # Try random swaps
        if not current_set:
            break
        
        u_idx = random.choice(list(current_set))
        
        # Pick a random triple to add (not in current_set)
        # Rejection sampling
        t_idx = random.randint(0, num_triples - 1)
        attempts = 0
        while present[t_idx] and attempts < 100:
            t_idx = random.randint(0, num_triples - 1)
            attempts += 1
        if present[t_idx]:
            continue
        
        # Check if swap is valid
        # We need to check if adding t_idx to (current_set - {u_idx}) creates a K(2,2,2)
        # A K(2,2,2) is violated if all 8 triples are present.
        # After swap, for a K containing t_idx:
        #   - If K also contains u_idx: after removing u_idx and adding t_idx, 
        #     the count for K changes by: -1 (remove u_idx) + 1 (add t_idx) = 0
        #     But we need to check if it becomes 8.
        #     Actually, let's think more carefully.
        #
        # Before swap: K has some count c.
        # If t_idx in K and u_idx in K:
        #   After swap: count stays c (remove u, add t)
        #   But we're checking if the NEW state has a violation.
        #   The new state has t_idx present, u_idx absent.
        #   For K to be violated in new state, all 8 must be present in new state.
        #   Since u_idx is absent from new state, K cannot be violated if u_idx in K.
        #
        # If t_idx in K and u_idx not in K:
        #   Before swap: K had c triples present (none of which is t_idx, since t_idx not in current_set)
        #   After swap: K has c + 1 triples present (t_idx added)
        #   K is violated if c + 1 == 8, i.e., c == 7.
        #
        # So the check is:
        # For each K containing t_idx:
        #   If u_idx in K: skip (K won't be violated since u_idx is removed)
        #   Else: if k_counts[K] == 7, then violation (since t_idx would make it 8)
        
        valid_swap = True
        for k_idx in triple_to_k222s[t_idx]:
            if u_idx in k222_list[k_idx]:
                continue
            if k_counts[k_idx] == 7:
                valid_swap = False
                break
        
        if valid_swap:
            # Perform swap
            present[u_idx] = False
            present[t_idx] = True
            current_set.remove(u_idx)
            current_set.add(t_idx)
            
            # Update counts
            for k_idx in triple_to_k222s[u_idx]:
                k_counts[k_idx] -= 1
            for k_idx in triple_to_k222s[t_idx]:
                k_counts[k_idx] += 1
            
            current_size = len(current_set)
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
    
    return list(best_set)

def _simulated_annealing(initial_set, all_triples, triple_to_k222s, k222_list, time_limit=30.0):
    """
    Simulated annealing for global optimization.
    Uses a counter array for efficient conflict checking.
    """
    num_triples = len(all_triples)
    num_k222s = len(k222_list)
    
    # Build initial state
    present = [False] * num_triples
    for idx in initial_set:
        present[idx] = True
    
    k_counts = [0] * num_k222s
    for idx in initial_set:
        for k_idx in triple_to_k222s[idx]:
            k_counts[k_idx] += 1
    
    current_size = len(initial_set)
    best_size = current_size
    best_set = set(initial_set)
    
    start_time = time.time()
    current_set = set(initial_set)
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.995
    
    T = T_start
    
    while time.time() - start_time < time_limit and T > T_end:
        # Try a random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random triple
            if current_size >= num_triples:
                T *= cooling_rate
                continue
            t_idx = random.randint(0, num_triples - 1)
            if present[t_idx]:
                T *= cooling_rate
                continue
            
            # Check if adding creates violation
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if k_counts[k_idx] == 7:
                    valid = False
                    break
            
            if valid:
                present[t_idx] = True
                current_set.add(t_idx)
                for k_idx in triple_to_k222s[t_idx]:
                    k_counts[k_idx] += 1
                current_size += 1
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
            else:
                # Reject (or accept with probability exp(-delta/T) but delta is infinite here)
                pass
                
        elif move_type < 0.6:
            # Remove a random triple
            if not current_set:
                T *= cooling_rate
                continue
            u_idx = random.choice(list(current_set))
            
            present[u_idx] = False
            current_set.remove(u_idx)
            for k_idx in triple_to_k222s[u_idx]:
                k_counts[k_idx] -= 1
            current_size -= 1
            
            # Always accept removals (they can't create violations)
            # But we might want to revert if it doesn't help... 
            # For SA, we accept based on energy change. Here energy = -size.
            # Removing decreases size, so energy increases (worse).
            # Accept with probability exp(-1/T)
            if random.random() > (1.0 / (1.0 + T)):
                # Revert
                present[u_idx] = True
                current_set.add(u_idx)
                for k_idx in triple_to_k222s[u_idx]:
                    k_counts[k_idx] += 1
                current_size += 1
        else:
            # Swap
            if not current_set:
                T *= cooling_rate
                continue
            
            u_idx = random.choice(list(current_set))
            
            t_idx = random.randint(0, num_triples - 1)
            attempts = 0
            while present[t_idx] and attempts < 50:
                t_idx = random.randint(0, num_triples - 1)
                attempts += 1
            if present[t_idx]:
                T *= cooling_rate
                continue
            
            # Check if swap is valid
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if u_idx in k222_list[k_idx]:
                    continue
                if k_counts[k_idx] == 7:
                    valid = False
                    break
            
            if valid:
                present[u_idx] = False
                present[t_idx] = True
                current_set.remove(u_idx)
                current_set.add(t_idx)
                
                for k_idx in triple_to_k222s[u_idx]:
                    k_counts[k_idx] -= 1
                for k_idx in triple_to_k222s[t_idx]:
                    k_counts[k_idx] += 1
                
                # Size unchanged, so no energy change. Always accept.
            else:
                # Could accept with small probability, but for simplicity reject
                pass
        
        T *= cooling_rate
    
    return list(best_set)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n = 16
    all_triples, triple_to_idx, triple_to_k222s, k222_list = _precompute_k222s(n)
    num_triples = len(all_triples)
    
    best_set = set()
    
    # Phase 1: Multiple greedy restarts
    while time.time() - start_time < deadline - 40.0:
        random.seed()
        order = list(range(num_triples))
        random.shuffle(order)
        
        g_set = _greedy_construct(all_triples, triple_to_k222s, k222_list, order)
        
        if len(g_set) > len(best_set):
            best_set = set(g_set)
    
    # Phase 2: Local search on best
    if time.time() - start_time < deadline - 20.0:
        ls_set = _local_search(list(best_set), all_triples, triple_to_k222s, k222_list, time_limit=15.0)
        if len(ls_set) > len(best_set):
            best_set = set(ls_set)
    
    # Phase 3: Simulated annealing
    if time.time() - start_time < deadline - 5.0:
        sa_time = min(25.0, deadline - 5.0 - (time.time() - start_time))
        if sa_time > 5.0:
            sa_set = _simulated_annealing(list(best_set), all_triples, triple_to_k222s, k222_list, time_limit=sa_time)
            if len(sa_set) > len(best_set):
                best_set = set(sa_set)
    
    # Final local search polish
    if time.time() - start_time < deadline - 2.0:
        final_ls = _local_search(list(best_set), all_triples, triple_to_k222s, k222_list, time_limit=3.0)
        if len(final_ls) > len(best_set):
            best_set = set(final_ls)
    
    return [list(all_triples[i]) for i in sorted(best_set)]

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Found {len(res)} triples in {t1-t0:.2f}s")
