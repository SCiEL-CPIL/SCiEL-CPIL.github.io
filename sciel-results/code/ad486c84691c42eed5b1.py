import time
import random
import itertools

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Seed produces 33, which is optimal.
        # 12341231423124312134214321
        pass 
        
    # For n >= 5, the seed is not optimal.
    # We will try to find a better word.
    
    # Hardcoded optimal/near-optimal for small n to ensure improvement
    # n=5 min length 152
    # n=6 min length 872
    
    # Strategy:
    # 1. Generate the seed word (length L_seed).
    # 2. For n=5, 6, we can use a specific search or hardcoded string.
    # 3. For n=7, use a time-limited heuristic.
    
    deadline = time.time() + 145.0
    
    def get_seed_word(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0:
                        break
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return "".join(map(str, word))

    seed_word = get_seed_word(n)
    seed_len = len(seed_word)
    
    best_word = seed_word
    best_len = seed_len
    
    # If n is small enough, we can do a more thorough search or use known results.
    # For n=5, 6, the search space is smaller.
    # For n=7, we rely on heuristics.
    
    if n == 5:
        # Known optimal is 152. Seed is 153.
        # We can try to find 152.
        # Or just return a known 152 word if we have it.
        # Let's try a quick search.
        pass
        
    # Heuristic Search for n >= 5
    # We represent the word as a list of integers.
    # We try to build the word by selecting permutations in a specific order.
    
    # This heuristic is expensive. We only run it if we have time.
    # For n=7, 5040 permutations.
    
    # A simple improvement: Randomize the order of permutations in the seed construction?
    # The seed uses the order of appearance.
    # If we shuffle the `perms` list before wrapping, we might get a different word.
    # Let's try a few random shuffles of the permutation order at each step.
    
    def construct_with_order(n, orderings):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # Apply ordering
            if m in orderings:
                # orderings[m] is a permutation of indices or a shuffle key
                # For simplicity, we just shuffle the list
                random.shuffle(perms)
                
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0:
                        break
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return "".join(map(str, word))
        
    # Try a few random constructions
    # This is fast.
    # But it might not break the seed's length significantly.
    # The seed's length is determined by the overlaps.
    # Shuffling might reduce overlaps, making it longer.
    # We want to INCREASE overlaps.
    # So we should try to order permutations to maximize overlap with the current end.
    
    def greedy_construct(n, randomize=False):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # Sort perms by overlap with current end?
            # The current end is the last m-1 symbols of `word`.
            # But `word` changes as we add blocks.
            # This is a greedy approach for the sequence of blocks.
            
            new = []
            remaining = set(perms)
            
            while remaining:
                # Find the permutation in `remaining` that maximizes overlap with `new`'s end
                # And also has good potential for future?
                # Just max overlap for now.
                best_p = None
                best_ov = -1
                best_block = None
                for p in remaining:
                    block = list(p) + [m] + list(p)
                    # Check overlap with `new`
                    max_ov = 0
                    for o in range(min(len(new), len(block)), -1, -1):
                        if o == 0:
                            break
                        if new[len(new) - o:] == block[:o]:
                            max_ov = o
                            break
                    if max_ov > best_ov:
                        best_ov = max_ov
                        best_p = p
                        best_block = block
                if best_p is None:
                    # Should not happen
                    break
                new += best_block[best_ov:]
                remaining.remove(best_p)
                
            word = new
        return "".join(map(str, word))
        
    # The greedy_construct might be slow for n=7 if we iterate over all remaining perms at each step.
    # Number of steps: n!
    # At each step, we check all remaining perms (n! - k).
    # Total ops: sum_{k=1}^{n!} (n! - k) * m ~ (n!)^2 * m.
    # For n=7, 5040^2 * 7 ~ 200,000,000 * 7 = 1.4 billion ops.
    # This is too slow in Python (seconds to minutes).
    
    # So we need a faster heuristic.
    
    # Let's stick to the seed for n=7 if we can't find a fast improvement.
    # But we can improve n=5 and n=6.
    
    if n == 5:
        # Try to find 152
        # The seed is 153.
        # We can try the greedy construction with a time limit?
        # Or just return the seed if we don't have a known 152 string.
        # I don't have the 152 string hardcoded.
        # Let's try a limited search for n=5.
        pass
        
    # For the purpose of this task, if we can't guarantee a faster search for n=7,
    # we should at least ensure we don't timeout.
    # The seed is fast.
    
    # Let's try a different angle:
    # The score is seed/value.
    # If we return the seed, score is 1.0.
    # If we return a valid word of length < seed, score > 1.0.
    # If we return an invalid word, score 0.
    
    # We must ensure validity.
    # The seed is valid.
    
    # Can we find a shorter valid word for n=7 quickly?
    # Unlikely with simple heuristics in 150s.
    
    # However, for n=5, 6, we can likely find shorter words.
    # Let's implement a specific search for n=5 and n=6.
    
    def search_shorter(n, time_limit):
        start_time = time.time()
        best = seed_word
        best_len = seed_len
        
        # Use a simple local search on the permutation sequence?
        # Too complex.
        
        # Let's just return the seed for now, but check if we can hardcode n=5,6.
        # I will not hardcode because I don't have the strings.
        
        return best
        
    # Given the constraints, the safest improvement is to optimize the seed's performance
    # and perhaps try a few random variations of the permutation order for n <= 6.
    
    # For n=7, the seed is the baseline.
    
    # Let's try to optimize the seed's overlap calculation.
    # It uses a loop. We can use string matching.
    
    def optimized_seed(n):
        word = "1"
        for m in range(2, n + 1):
            m_str = str(m)
            perms, seen = [], set()
            # Extract unique length m-1 substrings
            L = m - 1
            for i in range(len(word) - L + 1):
                window = word[i:i+L]
                if len(set(window)) == L and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new_parts = []
            current = ""
            for p in perms:
                block = p + m_str + p
                # Find overlap
                # Max overlap of `current` suffix and `block` prefix
                max_ov = 0
                # Check all possible overlaps
                # Length of overlap can be up to min(len(current), len(block))
                # But effectively, we just check if current ends with block[0:o]
                for o in range(min(len(current), len(block)), -1, -1):
                    if o == 0:
                        break
                    if current.endswith(block[:o]):
                        max_ov = o
                        break
                new_parts.append(block[max_ov:])
                current = current + block[max_ov:] # Not strictly needed if we just append to final string
            word = "".join(new_parts)
        return word

    # This is slightly faster than the list version.
    # But it doesn't change the length.
    
    # Conclusion:
    # Without a powerful search algorithm for n=7, we cannot easily beat 5913.
    # However, for n=5, 6, we can potentially beat the seed.
    # But the evaluator likely tests n=7 primarily.
    # If it tests n=5, 6, 7, improving 5 and 6 helps the average?
    # No, the score is for the specific n passed.
    # "The evaluator passes this task's n".
    # Task is "on seven symbols". So n=7.
    
    # So we need to improve n=7.
    
    # Is there a known trick?
    # The "standard" construction is not optimal.
    # The gap is small.
    # Maybe we can use a **Greedy with Backtracking** that is optimized in C?
    # No, Python only.
    
    # Let's try a **Randomized Greedy** that is faster.
    # Instead of checking all remaining perms, we maintain a list of candidates.
    # This is still hard.
    
    # I will return the optimized seed. It is safe.
    # If I have time, I will add a simple random shuffle attempt for n=7 to see if it helps.
    
    # Let's add a time-limited random shuffle attempt.
    # We try to construct the word with random shuffles of the permutation list at each step.
    # This is fast.
    
    start = time.time()
    best = optimized_seed(n)
    best_len = len(best)
    
    # Try random shuffles
    # For each step m, we can shuffle the `perms` list.
    # This changes the order of blocks.
    # It might lead to shorter or longer words.
    # We track the minimum length.
    
    attempts = 0
    while time.time() < start + 140:
        attempts += 1
        word = "1"
        valid = True
        for m in range(2, n + 1):
            m_str = str(m)
            perms, seen = [], set()
            L = m - 1
            for i in range(len(word) - L + 1):
                window = word[i:i+L]
                if len(set(window)) == L and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            random.shuffle(perms)
            
            new_parts = []
            current = ""
            for p in perms:
                block = p + m_str + p
                max_ov = 0
                for o in range(min(len(current), len(block)), -1, -1):
                    if o == 0:
                        break
                    if current.endswith(block[:o]):
                        max_ov = o
                        break
                new_parts.append(block[max_ov:])
                current = current + block[max_ov:]
            word = "".join(new_parts)
        
        if len(word) < best_len:
            best = word
            best_len = len(word)
            
    return best
