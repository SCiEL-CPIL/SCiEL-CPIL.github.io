"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

DEADLINE = time.time() + 82  # safety margin before 90s kill

def _precompute_gcds(n):
    nums = list(range(1, n + 1))
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(nums[i], nums[j])
            gcds[i][j] = g
            gcds[j][i] = g
    return nums, gcds

def _build_bad_triples(n, gcds):
    """
    Precompute all bad triples.
    Returns list of (i, j, k) indices (0-based) such that gcd(i,j)==gcd(i,k)==gcd(j,k)
    """
    bad = []
    for i in range(n):
        for j in range(i + 1, n):
            g_ij = gcds[i][j]
            for k in range(j + 1, n):
                if gcds[i][k] == g_ij and gcds[j][k] == g_ij:
                    bad.append((i, j, k))
    return bad

def _is_valid_set(S, n, bad_triples, in_set):
    """
    Check if the current set (represented by in_set boolean list) has no bad triples.
    We just check if any bad triple has all three in the set.
    """
    for (i, j, k) in bad_triples:
        if in_set[i] and in_set[j] and in_set[k]:
            return False
    return True

def _count_bad_triples_in_set(in_set, n, bad_triples):
    """
    Count how many bad triples are fully contained in the current set.
    """
    count = 0
    for (i, j, k) in bad_triples:
        if in_set[i] and in_set[j] and in_set[k]:
            count += 1
    return count

def _local_search_sa(in_set, n, bad_triples, rng, start_temp, end_temp, steps, deadline):
    """
    Simulated Annealing on the set of elements.
    State: in_set (list of booleans)
    Move: flip one bit (add or remove an element)
    Energy: -|S| + C * (number of bad triples in S)
    We want to minimize energy, i.e., maximize |S| - C * bad_triples_count.
    Actually, we want a valid set (bad_triples_count == 0) with max |S|.
    So energy = -|S| + BIG * bad_triples_count.
    """
    BIG = 1000  # Large penalty for invalidity
    
    def energy(in_set):
        size = sum(in_set)
        bad_count = _count_bad_triples_in_set(in_set, n, bad_triples)
        return -size + BIG * bad_count
    
    current_energy = energy(in_set)
    best_in_set = in_set[:]
    best_energy = current_energy
    
    # Precompute for each element x, the list of bad triples it is part of.
    # This speeds up the energy delta calculation.
    triples_by_element = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(bad_triples):
        triples_by_element[i].append(idx)
        triples_by_element[j].append(idx)
        triples_by_element[k].append(idx)
        
    # To check if a specific triple is "active" (all 3 in set), we just check in_set.
    # Delta energy when flipping x:
    # If x is added (in_set[x] was False -> True):
    #   size increases by 1 -> energy decreases by 1.
    #   bad_count increases by the number of triples containing x where the other 2 are in set.
    # If x is removed (in_set[x] was True -> False):
    #   size decreases by 1 -> energy increases by 1.
    #   bad_count decreases by the number of triples containing x where the other 2 are in set.
    
    def delta_energy(in_set, x):
        # x is the element to flip
        # Let's calculate the change in bad_count.
        # If in_set[x] is True, we are removing it.
        #   bad_count decreases by count of triples (x, y, z) where in_set[y] and in_set[z].
        # If in_set[x] is False, we are adding it.
        #   bad_count increases by count of triples (x, y, z) where in_set[y] and in_set[z].
        
        change_bad = 0
        for t_idx in triples_by_element[x]:
            i, j, k = bad_triples[t_idx]
            # The other two elements
            others = [y for y in (i, j, k) if y != x]
            y1, y2 = others[0], others[1]
            if in_set[y1] and in_set[y2]:
                change_bad += 1
        
        # If adding (was False), bad_count increases by change_bad.
        # If removing (was True), bad_count decreases by change_bad.
        
        if in_set[x]: # Removing
            # size -1 -> energy +1
            # bad_count -change_bad -> energy -BIG * change_bad
            return 1 - BIG * change_bad
        else: # Adding
            # size +1 -> energy -1
            # bad_count +change_bad -> energy +BIG * change_bad
            return -1 + BIG * change_bad

    current_in_set = in_set[:]
    current_e = energy(current_in_set)
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        # Choose a random element to flip
        x = rng.randint(0, n - 1)
        
        de = delta_energy(current_in_set, x)
        new_e = current_e + de
        
        if new_e < current_e or rng.random() < pow(2.71828, -de / max(temp, 1e-9)):
            current_in_set[x] = not current_in_set[x]
            current_e = new_e
            
            if current_e < best_energy:
                best_energy = current_e
                best_in_set = current_in_set[:]
                
    return best_in_set, best_energy

def construct():
    n = 128
    nums, gcds = _precompute_gcds(n)
    bad_triples = _build_bad_triples(n, gcds)
    
    # We want to find a subset S of [0, n-1] such that no bad triple is fully contained in S.
    # This is equivalent to finding a Maximum Independent Set in a 3-uniform hypergraph,
    # or a hitting set problem on the dual.
    # SA on the bits of S is a good approach.
    
    rng = random.Random(42)
    
    best_in_set = [False] * n
    best_size = 0
    
    # Multiple restarts with different initializations
    for restart in range(50):
        if time.time() > DEADLINE:
            break
            
        # Initialization: 
        # 1. Start with all True, then remove elements to fix conflicts?
        # 2. Start with random set?
        # 3. Start with greedy set?
        
        # Let's try a mix.
        # For first few restarts, try greedy.
        if restart < 10:
            # Greedy: add elements that don't create conflicts
            in_set = [False] * n
            order = list(range(n))
            rng.shuffle(order)
            for x in order:
                # Check if adding x creates a bad triple
                # x creates a bad triple if there exist y, z in S such that (x,y,z) is bad.
                # We can check this by iterating triples_by_element[x]
                # But we need triples_by_element here. Let's rebuild or pass it.
                # For simplicity in greedy, let's just check all pairs in S.
                # Or better, use the precomputed bad_triples.
                
                # Efficient check for adding x:
                # Is there any bad triple (x, y, z) where y, z are in S?
                # We can precompute for each x, the list of (y, z) pairs from bad triples.
                # pairs_for_x[x] = list of (y, z) such that (x,y,z) is bad.
                
                # Let's build this once outside the loop.
                pass
        else:
            # Random initialization
            in_set = [rng.random() < 0.5 for _ in range(n)]
            
        # Run SA
        # Start temp high, end temp low
        # Steps: enough to converge within time budget
        # Budget per restart: ~1.5s
        # Let's run 10000 steps.
        
        # We need pairs_for_x for efficient delta calculation in SA.
        # Let's build it now.
        pairs_for_x = [[] for _ in range(n)]
        for (i, j, k) in bad_triples:
            # For i, the pair is (j, k)
            pairs_for_x[i].append((j, k))
            pairs_for_x[j].append((i, k))
            pairs_for_x[k].append((i, j))
            
        # Redefine delta_energy to use pairs_for_x for speed
        def delta_energy_fast(in_set, x):
            change_bad = 0
            for (y, z) in pairs_for_x[x]:
                if in_set[y] and in_set[z]:
                    change_bad += 1
            if in_set[x]: # Removing
                return 1 - 1000 * change_bad
            else: # Adding
                return -1 + 1000 * change_bad

        # Modify SA to use fast delta
        # Actually, let's just implement the SA loop directly here with the fast delta.
        
        current_in_set = in_set[:]
        
        def energy_fast(in_set):
            size = sum(in_set)
            bad_count = 0
            for (i, j, k) in bad_triples:
                if in_set[i] and in_set[j] and in_set[k]:
                    bad_count += 1
            return -size + 1000 * bad_count
        
        current_e = energy_fast(current_in_set)
        best_in_set_local = current_in_set[:]
        best_e_local = current_e
        
        steps = 20000
        start_temp = 10.0
        end_temp = 0.01
        
        for step in range(steps):
            if time.time() > DEADLINE:
                break
                
            temp = start_temp * (end_temp / start_temp) ** (step / steps)
            x = rng.randint(0, n - 1)
            
            de = delta_energy_fast(current_in_set, x)
            new_e = current_e + de
            
            if new_e < current_e or rng.random() < pow(2.71828, -de / max(temp, 1e-9)):
                current_in_set[x] = not current_in_set[x]
                current_e = new_e
                
                if current_e < best_e_local:
                    best_e_local = current_e
                    best_in_set_local = current_in_set[:]
                    
        # Check if best_in_set_local is valid (should be, if energy is low enough)
        # If not valid, we might need to repair it or just reject it.
        # For now, if it's valid, update global best.
        if _is_valid_set(best_in_set_local, n, bad_triples, best_in_set_local):
            size = sum(best_in_set_local)
            if size > best_size:
                best_size = size
                best_in_set = best_in_set_local[:]
        else:
            # Try to repair: remove elements from bad triples
            # Simple repair: while there is a bad triple, remove a random element from it
            temp_set = best_in_set_local[:]
            changed = True
            while changed:
                changed = False
                for (i, j, k) in bad_triples:
                    if temp_set[i] and temp_set[j] and temp_set[k]:
                        # Remove one
                        idx = rng.choice([i, j, k])
                        temp_set[idx] = False
                        changed = True
                        break
            if _is_valid_set(temp_set, n, bad_triples, temp_set):
                size = sum(temp_set)
                if size > best_size:
                    best_size = size
                    best_in_set = temp_set[:]

    # Final check
    if best_size == 0:
        return [1]
        
    result = sorted(nums[i] for i in range(n) if best_in_set[i])
    return result
# EVOLVE-BLOCK-END
