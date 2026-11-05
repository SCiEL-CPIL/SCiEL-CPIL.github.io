"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations_with_replacement
import numpy as np

def construct():
    n = 4096
    
    # We want to maximize the number of elements.
    # The greedy approach in the seed picks small numbers.
    # A better strategy for B_h sequences often involves using larger steps or specific constructions.
    # However, the task asks for a CSP/Backtracking approach with heuristics.
    
    # Given the constraint that every unordered triple (with repetition) must have a unique sum,
    # this is a Sidon set of order 3 (or B_3 sequence).
    
    # The maximum size of a B_3 sequence in [1, N] is known to be roughly O(N^(1/3)).
    # For N=4096, N^(1/3) is about 16. But wait, the seed program returns up to 80?
    # Let's check the seed program's output size.
    # The seed program iterates x from 1 to 512 and adds it if no collision.
    # Let's estimate the size.
    # Actually, for B_3 sequences, the upper bound is around N^(1/3) * constant.
    # 4096^(1/3) = 16. So the size should be small, maybe 15-20?
    # Wait, the prompt says "Maximize cardinality... 1..80".
    # Let's run the seed logic mentally or check known results.
    # Known B_3 sequence in [1, 1000] has size ~10-12.
    # In [1, 4096], size might be ~16-20.
    
    # The seed program is a greedy algorithm. It might get stuck or produce a suboptimal set.
    # A backtracking approach with good heuristics can find larger sets.
    
    # However, implementing a full backtracking CSP for this might be slow if not careful.
    # The domain is [1, 4096].
    # If we use backtracking, we need to prune heavily.
    
    # Alternative: Use a known construction or a smarter greedy.
    # But the directive says: "Model the problem as a Constraint Satisfaction Problem... use a backtracking search with heuristic ordering".
    
    # Let's implement a backtracking search.
    # State: current set S, current set of sums T.
    # To speed up, we can use a bitset for sums.
    # Max sum = 3 * 4096 = 12288.
    # We can use a boolean array or a set of sums.
    
    # Heuristic: Try numbers in an order that is likely to work.
    # For Sidon sets, larger numbers are often better spaced.
    # Or we can use the "greedy with backtracking" approach.
    
    # Given the 60s limit, a deep backtracking might be risky.
    # Let's try a hybrid: Greedy with a limited backtracking depth or a randomized greedy.
    
    # Actually, for B_3 sequences, the greedy algorithm is often quite effective.
    # The seed program IS a greedy algorithm.
    # Why would backtracking improve it?
    # Greedy gets stuck if a small number blocks many future options.
    # Backtracking can undo that choice.
    
    # Let's implement a backtracking search that tries to add elements.
    # To make it fast, we'll use a recursive function with pruning.
    # We'll try to fill the set.
    
    # Optimization:
    # Instead of storing all sums in a list and checking collisions,
    # we maintain a set of used sums.
    
    # To improve on greedy, we can use a heuristic:
    # When choosing the next number, prefer numbers that leave the most options open.
    # This is expensive to compute.
    
    # A simpler improvement: Randomized Greedy.
    # Try multiple random orderings or random restarts.
    
    # But the directive specifically asks for CSP/Backtracking.
    
    # Let's implement a backtracking solver that tries to find a set of size K.
    # We can binary search on K, or just try to maximize.
    
    # Given the small expected size (~16-20), backtracking is feasible.
    
    # We will use a recursive backtracking function.
    # State: current_set, used_sums, next_start_index
    # We iterate through candidates from next_start_index to n.
    
    # To prune:
    # If we pick x, we generate new sums.
    # If any new sum is in used_sums, prune.
    
    # Heuristic:
    # We can try candidates in increasing order, but also allow skipping.
    # To maximize size, we want to pick numbers that are "compatible" with many others.
    
    # Let's try a different approach:
    # The seed program is O(K^3 * N) roughly.
    # K is small.
    
    # Let's implement a backtracking search that tries to build the set.
    # We'll use a global best solution.
    
    import time
    start_time = time.time()
    timeout = 55  # seconds
    
    best_set = []
    
    # Precompute all possible triples? No, too many.
    # We'll generate sums on the fly.
    
    def add_element(set_list, used_sums, x):
        """
        Try to add x to set_list.
        Returns (new_set, new_used_sums, success)
        """
        # Generate all new triples involving x
        # Triples are (a,b,c) with a<=b<=c
        # New triples are those where at least one element is x.
        # Since x is the largest (we build in increasing order? Not necessarily, but let's assume we add in increasing order for simplicity of generation, or we sort at the end).
        # Actually, if we don't enforce increasing order, we have to check all triples.
        # Let's enforce increasing order: x > last element in set_list.
        # This avoids permutations and ensures we generate each unordered triple exactly once when it's completed.
        
        if set_list and x <= set_list[-1]:
            return None, None, False
        
        new_set = set_list + [x]
        
        # Check collisions
        # Existing sums in used_sums.
        # New sums generated by triples containing x.
        
        # Triples containing x:
        # (x, x, x)
        # (a, x, x) for a in set_list
        # (a, b, x) for a <= b in set_list
        
        new_sums = set()
        
        # (x, x, x)
        s = 3 * x
        if s in used_sums or s in new_sums:
            return None, None, False
        new_sums.add(s)
        
        # (a, x, x)
        for a in set_list:
            s = a + 2 * x
            if s in used_sums or s in new_sums:
                return None, None, False
            new_sums.add(s)
        
        # (a, b, x) where a <= b
        # We need to iterate over pairs in set_list
        # Since set_list is sorted, we can do nested loops
        for i in range(len(set_list)):
            a = set_list[i]
            for j in range(i, len(set_list)):
                b = set_list[j]
                s = a + b + x
                if s in used_sums or s in new_sums:
                    return None, None, False
                new_sums.add(s)
        
        # If we get here, x is valid
        new_used_sums = used_sums | new_sums
        return new_set, new_used_sums, True
    
    def backtrack(set_list, used_sums, start_idx):
        nonlocal best_set
        
        if time.time() - start_time > timeout:
            return
        
        if len(set_list) > len(best_set):
            best_set = set_list[:]
        
        # Heuristic: Try to add more elements
        # We can stop if we've reached a good size, but we want to maximize.
        
        # Try candidates from start_idx to n
        # To improve efficiency, we can limit the range or use a heuristic.
        
        # A common heuristic for Sidon sets is to try larger numbers first?
        # Or smaller numbers first?
        # Smaller numbers pack tighter but collide more easily.
        # Larger numbers have more space.
        
        # Let's try increasing order first (like greedy), but with backtracking.
        
        for x in range(start_idx, n + 1):
            if time.time() - start_time > timeout:
                return
            
            new_set, new_used_sums, success = add_element(set_list, used_sums, x)
            if success:
                backtrack(new_set, new_used_sums, x + 1)
                
                # Pruning: If we've explored enough, we might break?
                # No, we need to explore other options to find larger sets.
            
            # Optional: If we find a solution of size K, and we know max possible is K, we can stop.
            # But we don't know max possible exactly.
    
    # Initialize
    backtrack([], set(), 1)
    
    return best_set

# EVOLVE-BLOCK-END
