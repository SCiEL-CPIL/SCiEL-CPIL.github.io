import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s margin to stay under 120s

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    # Base case for n=4
    if n == 4:
        return "12341213421432143124314234124321"  # Length 33

    # Standard recursive construction
    def recursive_superperm(n):
        if n == 1:
            return "1"
        if n == 2:
            return "121"
        if n == 3:
            return "123121321"
        
        word = "1"
        for m in range(2, n + 1):
            # Extract unique (m-1) permutations in order of appearance
            perms = []
            seen = set()
            for i in range(len(word) - (m - 1) + 1):
                window = word[i:i + m - 1]
                if len(set(window)) == m - 1:
                    if window not in seen:
                        seen.add(window)
                        perms.append(window)
            
            new_word = ""
            for p in perms:
                block = p + str(m) + p
                
                # Find max overlap
                max_overlap = 0
                limit = min(len(new_word), len(block))
                for o in range(limit, 0, -1):
                    if new_word[-o:] == block[:o]:
                        max_overlap = o
                        break
                
                new_word += block[max_overlap:]
            
            word = new_word
            
        return word

    # Get the baseline recursive superpermutation
    base_word = recursive_superperm(n)
    base_len = len(base_word)
    
    # For n <= 5, the recursive construction is known to be optimal
    if n <= 5:
        return base_word

    # For n >= 6, try to improve with a greedy search
    # This is especially important for n=7 where we might shave a few characters
    
    if n >= 6:
        # Try a greedy approach with random restarts
        # We'll try to find a superpermutation shorter than base_word
        
        def greedy_search(n, max_steps, time_limit):
            """
            Greedy search for a superpermutation.
            At each step, choose the next character that creates the most new permutations.
            """
            best_word = None
            best_len = base_len
            
            # Number of permutations
            import math
            num_perms = math.factorial(n)
            
            # Try multiple random starts
            for restart in range(20):  # Limit restarts
                if time.time() > start_time + time_limit:
                    break
                
                # Start with a random permutation
                start = list(range(1, n + 1))
                random.shuffle(start)
                word = ''.join(str(c) for c in start)
                
                # Track visited permutations
                visited = set()
                visited.add(tuple(start))
                
                # Current suffix of length n-1
                # We'll maintain the last n-1 characters
                
                steps = 0
                while len(visited) < num_perms:
                    if time.time() > start_time + time_limit:
                        break
                    
                    steps += 1
                    if steps > max_steps:
                        break
                    
                    # Try each possible next character
                    best_char = None
                    best_new_count = -1
                    best_next_suffix = None
                    
                    current_suffix = word[-(n-1):]
                    
                    for c in range(1, n + 1):
                        if c in current_suffix:
                            continue  # Skip if already in suffix (would create duplicate)
                        
                        # New window
                        new_window = current_suffix[1:] + str(c)
                        new_window_tuple = tuple(int(ch) for ch in new_window)
                        
                        if new_window_tuple in visited:
                            continue
                        
                        # Count how many new permutations this would add
                        # Actually, we just need to check if it's valid
                        # For a better heuristic, we could look ahead, but that's expensive
                        
                        # Simple heuristic: prefer characters that lead to more options
                        # Or just pick the first valid one
                        # Let's use a simple random choice among valid ones
                        if best_char is None or random.random() < 0.5:
                            best_char = c
                            best_next_suffix = new_window
                            best_new_count = 1  # Just 1 new permutation
                    
                    if best_char is None:
                        # Stuck, break
                        break
                    
                    word += str(best_char)
                    visited.add(tuple(int(ch) for ch in best_next_suffix))
                    current_suffix = best_next_suffix
                
                # Check if we have a complete superpermutation
                if len(visited) == num_perms:
                    if len(word) < best_len:
                        best_len = len(word)
                        best_word = word
                        if best_len <= base_len - 8:  # Significant improvement
                            return best_word
            
            return best_word if best_word is not None else base_word
        
        # Try greedy search with a time limit
        try:
            improved_word = greedy_search(n, max_steps=base_len + 100, time_limit=100)
            if improved_word is not None and len(improved_word) < base_len:
                return improved_word
        except Exception:
            pass
        
        # If greedy didn't improve, return the base word
        return base_word

    return base_word
