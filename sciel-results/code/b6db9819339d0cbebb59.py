import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s buffer for 60s limit
    
    # We want to return a list of ints
    # Initial best known valid set (from seed or simple construction)
    # The seed program had [1, 2, 4, 8, 16, 32, 64, 128, 256] which is size 9.
    # We will try to find a larger one.
    
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best_set)
    
    # Candidates are 1..256
    # We try to build the set by adding elements.
    # Heuristic: Larger elements first to separate sum layers.
    candidates = list(range(256, 0, -1))
    
    # sums_by_k[k] is a set of sums for subsets of size k
    # Initially, with empty set, sums_by_k = [{0}, set(), ...]
    # We start with empty set.
    
    def check_valid(sums_by_k, max_k):
        """
        Check if sums_by_k[0..max_k] are pairwise disjoint.
        """
        for i in range(max_k + 1):
            for j in range(i + 1, max_k + 1):
                if not sums_by_k[i].isdisjoint(sums_by_k[j]):
                    return False
        return True

    def dfs(current_set, sums_by_k, max_k, candidates):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
            
        if not candidates:
            return
            
        # Try adding each candidate
        # Optimization: We can break early if we think we can't beat best_size?
        # No, we just need to find ANY set larger than best_size.
        
        for i, v in enumerate(candidates):
            if time.time() > deadline:
                return
                
            # Compute new sums if we add v
            # new_sums_by_k will have size max_k + 2 (indices 0 to max_k+1)
            new_sums_by_k = [set() for _ in range(max_k + 2)]
            
            # Copy old sums
            # S'_k = S_k U {s + v | s in S_{k-1}}
            # Note: S_0 is always {0}
            
            # We can do this in place on a copy or build new list
            # Building new list is safer
            
            for k in range(max_k + 1):
                # Existing sums of size k
                new_sums_by_k[k].update(sums_by_k[k])
                
            # Add new sums involving v
            for k in range(1, max_k + 2):
                # Sums of size k-1 from old set, plus v
                prev_sums = sums_by_k[k-1]
                for s in prev_sums:
                    new_sums_by_k[k].add(s + v)
                    
            # Check validity
            if check_valid(new_sums_by_k, max_k + 1):
                # Recurse
                new_current_set = current_set + [v]
                new_candidates = candidates[:i] + candidates[i+1:]
                dfs(new_current_set, new_sums_by_k, max_k + 1, new_candidates)
                
        # Note: We don't backtrack by removing from current_set because we passed copies/immutable lists
        # current_set is a list, we create new list in recursion.
        
    # Initial state: empty set
    # sums_by_k[0] = {0}, others empty
    initial_sums = [set() for _ in range(1)]
    initial_sums[0].add(0)
    
    dfs([], initial_sums, 0, candidates)
    
    # Try global conflict-driven removal approach
    # Start with a random large set and remove elements involved in the most frequent cross-cardinality sum collisions
    
    def compute_sums_by_cardinality(s):
        """
        Compute sums_by_k for a given set s.
        s is a list of integers.
        Returns a list of sets: sums_by_k[k] is the set of all subset sums of size k.
        """
        n = len(s)
        # Use DP: sums_by_k[k] for k in 0..n
        # Initialize
        sums_by_k = [set() for _ in range(n + 1)]
        sums_by_k[0].add(0)
        
        # Iterate over elements
        for v in s:
            # Update in reverse order to avoid using the same element multiple times
            for k in range(n, 0, -1):
                for prev_sum in sums_by_k[k-1]:
                    sums_by_k[k].add(prev_sum + v)
        
        return sums_by_k

    def find_collisions(sums_by_k, max_k):
        """
        Find cross-cardinality sum collisions.
        Returns a dict: element -> count of collisions it's involved in.
        Actually, we need to map collisions to elements.
        
        A collision is a sum value that appears in both sums_by_k[i] and sums_by_k[j] for i != j.
        For each such sum, we need to identify which elements could be responsible.
        
        Alternative: For each pair (i, j) with i < j, find common sums.
        For each common sum s, we can try to determine which elements are "responsible".
        This is complex. A simpler heuristic: 
        For each element v, count how many pairs of subsets (A, B) with |A| != |B| have sum(A) == sum(B) 
        and v is in A XOR B? Or just v is in A or v is in B?
        
        Let's use a simpler metric: 
        For each element v, compute the number of sums in sums_by_k that involve v.
        Actually, let's just count for each element v, the number of "conflicting pairs" of subsets
        where v is present in at least one of the subsets.
        
        Even simpler: 
        For each pair of cardinalities (i, j), i < j, let C_ij = sums_by_k[i] & sums_by_k[j].
        For each sum s in C_ij, we know there exists A subset of size i with sum s, and B subset of size j with sum s.
        The elements in A U B are "involved". 
        This is still hard to compute exactly without enumerating subsets.
        
        Let's try a different approach: 
        For each element v, compute a "conflict score" based on how often v appears in subsets that contribute to collisions.
        
        Alternative heuristic: 
        For each element v, remove it and see if the number of collisions decreases.
        This is O(n * cost_of_check).
        
        Let's try: For each element v in the current set, temporarily remove it, compute the number of collisions,
        and pick the v whose removal reduces collisions the most (or increases the "validity margin").
        
        Cost: O(n * n^2 * 2^n) worst case? No, sums_by_k computation is O(n^2 * 2^n) in worst case if we enumerate all subsets,
        but we are using DP on sums, which is O(n * max_sum * n) roughly?
        
        Actually, the DP for sums_by_k is:
        For each element v (n elements), for each k (up to n), for each sum in sums_by_k[k-1] (up to n * max_val), add v.
        So it's roughly O(n^2 * max_val) where max_val is the max possible sum (n * 256).
        For n=30, max_val = 7680, n^2 = 900, so 900 * 7680 ~ 7 million operations. This is feasible for a few iterations.
        
        Let's implement:
        1. Compute current number of collisions.
        2. For each element v, remove it, compute new number of collisions.
        3. Pick the v that gives the largest reduction in collisions.
        4. If no reduction, pick the one that gives the smallest increase, or stop.
        """
        n = len(sums_by_k) - 1 # max cardinality
        total_collisions = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                total_collisions += len(sums_by_k[i] & sums_by_k[j])
        
        return total_collisions

    def global_conflict_removal(initial_set, time_limit):
        """
        Start with initial_set, iteratively remove the element whose removal
        reduces the total number of cross-cardinality collisions the most.
        """
        current_set = list(initial_set)
        
        # Compute initial sums
        sums_by_k = compute_sums_by_cardinality(current_set)
        current_collisions = find_collisions(sums_by_k, len(current_set))
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_v = None
            best_collisions = current_collisions
            best_new_set = None
            best_new_sums = None
            
            n = len(current_set)
            if n == 0:
                break
                
            # For each element, try removing it
            for idx, v in enumerate(current_set):
                new_set = current_set[:idx] + current_set[idx+1:]
                new_sums = compute_sums_by_cardinality(new_set)
                new_collisions = find_collisions(new_sums, len(new_set))
                
                if new_collisions < best_collisions:
                    best_collisions = new_collisions
                    best_v = v
                    best_new_set = new_set
                    best_new_sums = new_sums
                    improved = True
                    
            if improved:
                current_set = best_new_set
                sums_by_k = best_new_sums
                current_collisions = best_collisions
                
        return current_set, current_collisions

    # Try global conflict removal starting from different initial sets
    # 1. Start from the best set found by DFS
    # 2. Start from random large sets
    
    # First, let's try to extend the best set found by DFS using a greedy approach
    # or try global conflict removal on a larger random set.
    
    # Let's try starting with a random set of size 20-30 and see if we can reduce collisions to 0.
    # If we can get to 0 collisions, we have a valid set.
    
    random.seed(42)
    
    # Try several random starting sets
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
            
        # Random set of size 20 to 30
        size = random.randint(20, 30)
        random_set = random.sample(range(1, 257), size)
        
        # Run global conflict removal
        result_set, final_collisions = global_conflict_removal(random_set, deadline - 5)
        
        if final_collisions == 0 and len(result_set) > best_size:
            best_set = result_set
            best_size = len(result_set)
            
    # Also try starting from the best DFS set and see if we can add more elements
    # using a greedy approach guided by collision count
    
    # For now, just return the best found
    # Ensure types are correct
    return [int(x) for x in best_set]
