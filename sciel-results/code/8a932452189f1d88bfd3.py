# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square modulo 205.
    Uses a combination of greedy initialization, local search (hill climbing), and simulated annealing
    to maximize the size of the valid set.
    """
    m = 205
    k = 2
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # Precompute conflict matrix: conflict[i][j] is True if (i-j)%m is in forbidden
    # Since (i-j)%m is in forbidden iff (j-i)%m is in forbidden (because -x^2 mod m is also a square? No.
    # Wait. F = {x^2 mod m}. Is F symmetric? i.e. if d in F, is -d in F?
    # For m=205=5*41.
    # Let's just build the matrix explicitly.
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            d = (i - j) % m
            if d in forbidden:
                conflict[i][j] = True
    
    # Also precompute for each node, the set of nodes it conflicts with
    # neighbors[i] = set of j such that conflict[i][j] is True
    neighbors = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if conflict[i][j]:
                neighbors[i].add(j)
    
    def is_valid(set_indices):
        """Check if the set of indices is independent (no conflicts)."""
        s = set(set_indices)
        for i in s:
            for j in neighbors[i]:
                if j in s:
                    return False
        return True

    def count_conflicts(set_indices):
        """Count number of conflicting pairs in the set."""
        s = set(set_indices)
        count = 0
        for i in s:
            for j in neighbors[i]:
                if j in s and i < j: # Count each pair once
                    count += 1
        return count

    # Strategy 1: Greedy Randomized Construction (multiple restarts)
    best_set = set()
    best_size = 0
    
    deadline = time.time() + 580  # Safety margin
    
    # 1. Simple Greedy (deterministic)
    current_set = set()
    for i in range(m):
        if all((i - j) % m not in forbidden for j in current_set):
            current_set.add(i)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)

    # 2. Randomized Greedy (multiple iterations)
    for restart in range(500):
        if time.time() > deadline:
            break
        random.shuffle_order = list(range(m))
        random.shuffle(random.shuffle_order)
        current_set = set()
        for i in random.shuffle_order:
            if all((i - j) % m not in forbidden for j in current_set):
                current_set.add(i)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)

    # 3. Local Search / Hill Climbing starting from best_set
    # We can try to add a node and remove conflicting nodes if it increases size (unlikely)
    # Or swap: remove one, add another.
    # Better: Start with a set, try to add any node that doesn't conflict.
    # If no node can be added, try to remove one node and see if we can add two.
    
    # Let's implement a more robust local search:
    # State: current set S
    # Move: Add a node v not in S such that no conflict. (Growth)
    # Move: Remove a node u in S, then try to add as many non-conflicting nodes as possible. (Shake)
    
    # Actually, since the graph is relatively dense, let's use a Max Independent Set heuristic.
    # Tomita's algorithm or similar branch and bound might be too slow for 205 nodes if not optimized, 
    # but 205 is small enough for a well-optimized B&B or heavy local search.
    
    # Let's try a simple Simulated Annealing on the set of selected nodes.
    # State: binary vector x of length m.
    # Objective: maximize sum(x) - penalty * conflicts(x)
    # We want 0 conflicts.
    
    # Initialize with best_set
    x = np.zeros(m, dtype=bool)
    for i in best_set:
        x[i] = True
    
    current_size = np.sum(x)
    current_conflicts = count_conflicts(np.where(x)[0])
    
    T = 1.0
    T_min = 1e-4
    alpha = 0.999
    
    steps = 0
    max_steps = 1000000
    
    while T > T_min and steps < max_steps:
        if time.time() > deadline:
            break
        
        # Pick a random move: flip a random bit
        i = random.randint(0, m - 1)
        
        # Calculate change in size and conflicts
        # If x[i] is True, we are removing i.
        # If x[i] is False, we are adding i.
        
        # Efficient conflict update:
        # Let C be the number of conflicts.
        # Let N_i be the neighbors of i.
        # If adding i:
        #   New conflicts = C + number of neighbors of i already in S
        # If removing i:
        #   New conflicts = C - number of neighbors of i already in S
        
        # Count neighbors in S
        # We can precompute neighbor lists as arrays for speed?
        # neighbors[i] is a set.
        # Let's just count.
        
        if x[i]:
            # Removing i
            # Size decreases by 1
            delta_size = -1
            # Conflicts decrease by number of neighbors in S
            # Note: conflicts are counted as pairs.
            # If i is in S, it conflicts with k neighbors in S.
            # Removing i removes these k conflicts.
            delta_conflicts = -len(neighbors[i] & set(np.where(x)[0]))
            
            new_size = current_size + delta_size
            new_conflicts = current_conflicts + delta_conflicts
            
            if new_conflicts < 0: # Should not happen
                new_conflicts = 0
            
            # Acceptance probability
            if new_conflicts < current_conflicts or (new_conflicts == current_conflicts and new_size > current_size):
                accept = True
            else:
                # Energy change: We prefer lower conflicts, then higher size.
                # Let's define Energy = conflicts * 1000 - size
                # delta_E = delta_conflicts * 1000 - delta_size
                delta_E = delta_conflicts * 1000 - delta_size
                if delta_E < 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta_E / T)
            
            if accept:
                x[i] = False
                current_size = new_size
                current_conflicts = new_conflicts
                
                if current_conflicts == 0 and current_size > best_size:
                    best_size = current_size
                    best_set = set(np.where(x)[0])
                    
        else:
            # Adding i
            delta_size = 1
            # Conflicts increase by number of neighbors in S
            delta_conflicts = len(neighbors[i] & set(np.where(x)[0]))
            
            new_size = current_size + delta_size
            new_conflicts = current_conflicts + delta_conflicts
            
            if new_conflicts < current_conflicts or (new_conflicts == current_conflicts and new_size > current_size):
                accept = True
            else:
                delta_E = delta_conflicts * 1000 - delta_size
                if delta_E < 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta_E / T)
            
            if accept:
                x[i] = True
                current_size = new_size
                current_conflicts = new_conflicts
                
                if current_conflicts == 0 and current_size > best_size:
                    best_size = current_size
                    best_set = set(np.where(x)[0])
        
        T *= alpha
        steps += 1

    # Final check
    if not is_valid(list(best_set)):
        # Fallback to a known valid set if something went wrong
        # Greedy again
        current_set = set()
        for i in range(m):
            if all((i - j) % m not in forbidden for j in current_set):
                current_set.add(i)
        best_set = current_set
        best_size = len(best_set)
        
    return np.array(sorted(list(best_set)), dtype=np.int64)

# EVOLVE-BLOCK-END
