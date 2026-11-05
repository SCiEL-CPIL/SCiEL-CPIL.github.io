"""
Strategy: Exact Maximum Independent Set (MIS) via Bit-Parallel Branch and Bound.
We model the problem as finding the largest subset of vertices in a conflict graph 
where an edge exists between two numbers if adding them to a set would create a 4-AP 
with existing elements? No, that's not a static graph. 
Wait, the property "contains no 4-AP" is not a simple pairwise conflict. 
A set S is valid if there is no {a, a+d, a+2d, a+3d} subset of S.
This is a hypergraph independent set problem (4-uniform).
Standard MIS solvers work on graphs. We can reduce 4-uniform hypergraph independent set 
to graph MIS by transforming the hypergraph?
Actually, for 4-APs, the constraints are local.
However, implementing a general hypergraph MIS solver is complex.
Let's look at the structure.
A 4-AP is determined by its first two terms (a, b) with d=b-a.
The conflict is between the set of all 4-tuples.

Alternative Exact Approach:
Since n=512 is small, we can use a specialized backtracking with strong pruning.
But "Exact" might be too slow for 512 if the density is high.
The greedy approach got ~1.0 (which is likely around 100-150 elements? 
Actually, R(4, 4) bounds suggest density is low. 
For 3-AP, max size is ~n/2. For 4-AP, it's lower.
Known constructions: Behrend's set gives ~n / (log n)^c.
For n=512, log n ~ 6.2. 
If we use the "digits in base 3" construction (Salem-Spencer), it gives 3-AP free sets. 
4-AP free sets are smaller.

Let's stick to the **Local Search / Simulated Annealing** but with a **much better initial state** 
and a **faster conflict check** using bitsets, but the prompt asks for a MATERIAL difference.
The current program is Greedy + Local Search.
Inspiration 3 suggested MIS.
Let's try a **Genetic Algorithm / Evolutionary Strategy** or **Simulated Annealing with Bitsets**.
The key bottleneck in the current code is the Python-level loops for conflict checking.
Using NumPy bitsets or Python integers as bitsets can speed this up by orders of magnitude.

NEW STRATEGY: 
1. Represent the set as a Python integer bitmask (512 bits).
2. Precompute for each possible 4-AP a bitmask of its elements? No, that's too many.
3. Instead, for each element x, precompute a bitmask `conflict_mask[x]` such that 
   if `current_set & conflict_mask[x]` has enough bits, it forms a 4-AP?
   No, a 4-AP requires 4 specific elements.
   
   Let's precompute for each x, the list of 4-APs containing x.
   There are O(n^2) 4-APs total.
   Total 4-APs in [1, 512]:
   d=1: 509
   d=2: 506
   ...
   Sum_{d=1}^{170} (512 - 3d) ~ 170*512 - 3*170*171/2 ~ 87000 - 43500 ~ 43000 4-APs.
   
   We can represent the set of "active" 4-APs as a bitmask? No, 43000 bits is fine (5375 bytes).
   But checking if *any* 4-AP is fully contained in the set is the goal.
   
   Better: 
   Maintain a count for each 4-AP of how many of its elements are in the current set.
   If we add x, we increment the count for all 4-APs containing x.
   If any count reaches 4, it's invalid.
   If we remove x, we decrement.
   
   This allows O(1) average cost per add/remove if we just track the max count?
   No, we need to know if *any* count is 4.
   We can maintain a counter `num_ap_with_4_elements`.
   When adding x:
     For each AP containing x:
       count[ap] += 1
       if count[ap] == 4: num_ap_with_4_elements += 1
   If num_ap_with_4_elements == 0, the set is valid.
   
   This is extremely fast!
   Number of APs containing a specific x is O(n).
   So add/remove is O(n) ~ 500 operations.
   This is 100x faster than the current O(n) set lookups with Python overhead.
   
   With this speedup, we can run **Simulated Annealing** or **Random Restart Hill Climbing** 
   for much longer, exploring the space more thoroughly.
   We will use **Simulated Annealing** to escape local optima, which the current greedy approach fails at.
"""

import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n_max = 512
    elements = list(range(1, n_max + 1))
    
    # Precompute all 4-term arithmetic progressions
    # ap_list[i] is a tuple of 4 elements (a, b, c, d)
    # We also need for each element x, the list of indices of APs containing x
    ap_list = []
    element_to_aps = {x: [] for x in range(1, n_max + 1)}
    
    # Generate all 4-APs: a, a+d, a+2d, a+3d
    # a >= 1, a+3d <= 512
    for d in range(1, (n_max - 1) // 3 + 1):
        max_a = n_max - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap = (a, b, c, e)
            idx = len(ap_list)
            ap_list.append(ap)
            for val in ap:
                element_to_aps[val].append(idx)
                
    num_aps = len(ap_list)
    
    # State representation:
    # in_set: boolean array or set of elements currently in the set
    # ap_counts: array of length num_aps, count of elements in each AP that are in_set
    # num_full_aps: count of APs where ap_counts[i] == 4
    
    # We will use Simulated Annealing.
    # Initial state: Random valid set.
    # We can start with an empty set and greedily add random elements to get a decent start.
    
    def get_initial_set():
        in_set = [False] * (n_max + 1)
        ap_counts = [0] * num_aps
        num_full_aps = 0
        
        # Greedy random initialization
        order = list(range(1, n_max + 1))
        random.shuffle(order)
        
        for x in order:
            # Check if adding x creates a full AP
            # We need to check if any AP containing x has count 3.
            # If so, adding x makes it 4.
            can_add = True
            for ap_idx in element_to_aps[x]:
                if ap_counts[ap_idx] == 3:
                    can_add = False
                    break
            
            if can_add:
                in_set[x] = True
                for ap_idx in element_to_aps[x]:
                    ap_counts[ap_idx] += 1
                    if ap_counts[ap_idx] == 4:
                        num_full_aps += 1
        
        return in_set, ap_counts, num_full_aps

    def add_element(x, in_set, ap_counts, num_full_aps):
        # Returns new num_full_aps
        in_set[x] = True
        new_num_full = num_full_aps
        for ap_idx in element_to_aps[x]:
            ap_counts[ap_idx] += 1
            if ap_counts[ap_idx] == 4:
                new_num_full += 1
        return new_num_full

    def remove_element(x, in_set, ap_counts, num_full_aps):
        # Returns new num_full_aps
        in_set[x] = False
        new_num_full = num_full_aps
        for ap_idx in element_to_aps[x]:
            ap_counts[ap_idx] -= 1
            if ap_counts[ap_idx] == 3: # Was 4, now 3
                new_num_full -= 1
        return new_num_full

    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.999 # Cooling rate
    
    # We will run multiple restarts to find the global optimum
    best_size = 0
    best_set = [False] * (n_max + 1)
    
    while time.time() < deadline:
        # 1. Initialize
        in_set, ap_counts, num_full_aps = get_initial_set()
        current_size = sum(in_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = in_set.copy()
        
        T = T_start
        
        # 2. Annealing Loop
        # We perform swaps (remove one, add one) to maintain size, 
        # or add/remove to change size.
        # To maximize size, we should bias towards adds.
        # But if we are stuck, we need to remove to unblock.
        
        # Let's use a move set:
        # 1. Add a random element not in set (if valid)
        # 2. Remove a random element in set
        # 3. Swap: Remove one, Add another (always valid if we check)
        
        # Actually, simpler: 
        # At each step, pick a random move.
        # If move increases size and is valid -> accept.
        # If move maintains size (swap) and is valid -> accept if better? No, size is same.
        # We want to maximize size. So we primarily want to add.
        # But we get stuck when no add is possible.
        # Then we must remove something to enable more adds later.
        
        # Strategy:
        # While no_improvement < limit:
        #   Try to add a random missing element.
        #   If valid: Add it.
        #   Else: Remove a random present element (random restart of local search within this SA run).
        
        # Let's do a simpler Hill Climbing with Random Restarts, but faster.
        # The speedup allows 10x more restarts.
        
        no_improvement = 0
        max_no_improvement = 1000 # Increased due to speed
        
        while time.time() < deadline and no_improvement < max_no_improvement:
            # Try to add
            missing = [x for x in range(1, n_max + 1) if not in_set[x]]
            if missing:
                x = random.choice(missing)
                # Check validity
                valid = True
                for ap_idx in element_to_aps[x]:
                    if ap_counts[ap_idx] == 3:
                        valid = False
                        break
                
                if valid:
                    num_full_aps = add_element(x, in_set, ap_counts, num_full_aps)
                    current_size += 1
                    no_improvement = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_set = in_set.copy()
                else:
                    no_improvement += 1
            else:
                # Set is full? Unlikely.
                break
            
            # Occasionally, to escape local optima, we do a "kick":
            # Remove k random elements.
            if no_improvement > 50 and random.random() < 0.1:
                # Remove 1-3 random elements
                present = [x for x in range(1, n_max + 1) if in_set[x]]
                if present:
                    k = random.randint(1, 3)
                    for _ in range(k):
                        if present:
                            y = random.choice(present)
                            present.remove(y)
                            num_full_aps = remove_element(y, in_set, ap_counts, num_full_aps)
                            current_size -= 1
                    no_improvement = 0 # Reset counter to allow new exploration

    # Convert best_set to list
    result = [i for i in range(1, n_max + 1) if best_set[i]]
    return result

# To ensure we don't crash if time runs out mid-loop, we return the best found so far.
# The loop checks time.time() < deadline.

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(len(res))
