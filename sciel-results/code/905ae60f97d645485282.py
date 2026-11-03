import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    if n <= 1:
        return "1"
    
    # For small n, we can use known optimal or simple construction
    # n=1: "1" (len 1)
    # n=2: "121" (len 3) -> 1!+2! = 3
    # n=3: "123121321" (len 9) -> 1!+2!+3! = 9
    # n=4: len 33
    # n=5: len 153
    # n=6: len 873
    # n=7: target 5905
    
    target_perms = n ** 0 # placeholder
    # Calculate n!
    fact = 1
    for i in range(1, n + 1):
        fact *= i
    
    # Initial state: permutation of 1..n-1
    # We start with 1, 2, ..., n-1
    current_state = tuple(range(1, n))
    word = list(current_state)
    
    # The first permutation is formed when we add the first symbol to the state
    # But the definition says "contains every permutation as n consecutive symbols".
    # The word starts with n-1 symbols. The first n-symbol window appears after the first append.
    # So we need to cover all n! permutations.
    
    covered = set()
    # Note: The initial state itself is not a permutation of 1..n.
    # The permutations are windows of length n.
    
    # We will simulate the path.
    # Each step appends a symbol x not in current_state.
    # The new window is current_state + (x,).
    
    # To make it fast, we can precompute all permutations of 1..n?
    # 5040 is small.
    all_perms = set()
    # Generating all perms of 1..n
    # Using itertools.permutations
    import itertools
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    best_word = None
    best_len = float('inf')
    
    # We will run a greedy search with random restarts if needed, 
    # but a single good greedy run might suffice.
    # Let's try a few iterations with different random seeds for tie-breaking.
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        current_state = tuple(range(1, n))
        word = list(current_state)
        covered = set()
        
        # The first permutation is not yet formed.
        # We need to add symbols until we have covered all n! permutations.
        
        # To allow for different paths, we can randomize the initial state?
        # No, the structure is symmetric. But the order of visiting matters.
        # We can randomize the tie-breaking.
        
        while len(covered) < fact:
            if time.time() > deadline:
                break
                
            candidates = []
            # Possible next symbols are those not in current_state
            for x in range(1, n + 1):
                if x not in current_state:
                    new_state = current_state[1:] + (x,)
                    new_perm = current_state + (x,)
                    
                    # Check if new_perm is covered
                    is_new = new_perm not in covered
                    
                    # Heuristic: Count how many outgoing edges from new_state lead to uncovered perms
                    # This is a lookahead of 1 step.
                    # Outgoing edges from new_state correspond to symbols y not in new_state.
                    # The perms would be new_state + (y,).
                    # We count how many of these are NOT in covered.
                    
                    uncovered_out = 0
                    for y in range(1, n + 1):
                        if y not in new_state:
                            perm_y = new_state + (y,)
                            if perm_y not in covered:
                                uncovered_out += 1
                    
                    # Priority:
                    # 1. Prefer moves that cover a new permutation (is_new)
                    # 2. Among those, prefer moves with more uncovered outgoing options (uncovered_out)
                    # 3. Random tie-breaker
                    
                    score = (1 if is_new else 0, uncovered_out, random.random())
                    candidates.append((score, x, new_state, new_perm))
            
            if not candidates:
                # Should not happen for n>=2
                break
            
            # Sort by score descending
            candidates.sort(key=lambda item: item[0], reverse=True)
            
            # Pick the best
            _, x, new_state, new_perm = candidates[0]
            
            word.append(x)
            covered.add(new_perm)
            current_state = new_state
            
        # Check if we finished
        if len(covered) == fact:
            current_len = len(word)
            if current_len < best_len:
                best_len = current_len
                best_word = word.copy()
                
            # If we found the theoretical lower bound or a very good length, we can stop?
            # Theoretical lower bound is not tight. Optimal for n=7 is 5905.
            # If we hit 5905, we are done.
            if best_len <= 5905:
                break

    if best_word:
        return "".join(map(str, best_word))
    else:
        # Fallback: Return the classic recursive construction if greedy failed or timed out
        # This ensures we return a valid (though possibly longer) result
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
                overlap = 0
                max_o = min(len(new), len(block))
                for o in range(max_o, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

# Keep the function name and signature
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program has `def construct(n):`.

# I should wrap the logic inside `construct` or define helper functions outside.
# The code above defines `construct` directly.

# One issue: `itertools.permutations` for n=7 generates 5040 items. This is fast.
# The inner loop `for y in range(1, n+1)` runs 7 times.
# The main loop runs ~5900 times.
# Total operations ~ 5900 * 7 * 7 = ~300,000. This is very fast in Python.
# So we can run many attempts.

# Let's refine the loop to run as many attempts as possible within the deadline.

# Rewriting construct to be more robust with the deadline.

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if n <= 1:
        return "1"
    
    import itertools
    
    fact = 1
    for i in range(1, n + 1):
        fact *= i
        
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    best_word = None
    best_len = float('inf')
    
    # For n=7, optimal is 5905.
    # We try to find a path of length 5905.
    
    # We will use a greedy algorithm with randomized tie-breaking.
    # We will run multiple iterations.
    
    for attempt in range(1000):
        if time.time() > deadline - 5.0: # Stop 5s early to ensure return
            break
            
        # Randomize the initial state? 
        # The graph is vertex-transitive in some sense, but let's just fix it to 1..n-1
        # and randomize the choice among equal-score candidates.
        
        current_state = tuple(range(1, n))
        word = list(current_state)
        covered = set()
        
        # Optimization: Use a local variable for covered to avoid global lookup
        cov = covered
        
        while len(cov) < fact:
            if time.time() > deadline - 1.0:
                break
                
            candidates = []
            # Possible next symbols
            # We can precompute the set of symbols not in current_state
            # But n is small, just iterate
            
            for x in range(1, n + 1):
                if x not in current_state:
                    new_state = current_state[1:] + (x,)
                    new_perm = current_state + (x,)
                    
                    is_new = new_perm not in cov
                    
                    # Lookahead: count uncovered outgoing from new_state
                    uncovered_out = 0
                    # Optimize: check if new_perm is new first? 
                    # If not new, the move is bad, but we might be forced.
                    
                    for y in range(1, n + 1):
                        if y not in new_state:
                            if (new_state + (y,)) not in cov:
                                uncovered_out += 1
                                
                    # Score:
                    # Primary: is_new (1 if new, 0 if old)
                    # Secondary: uncovered_out
                    # Tertiary: random
                    score = (1 if is_new else 0, uncovered_out, random.random())
                    candidates.append((score, x, new_state, new_perm))
            
            if not candidates:
                break
            
            # Sort and pick best
            # For small n, sorting 6-7 items is fast
            candidates.sort(key=lambda item: item[0], reverse=True)
            
            _, x, new_state, new_perm = candidates[0]
            
            word.append(x)
            cov.add(new_perm)
            current_state = new_state
            
        if len(cov) == fact:
            current_len = len(word)
            if current_len < best_len:
                best_len = current_len
                best_word = word.copy()
                
            if best_len <= 5905:
                break

    if best_word:
        return "".join(map(str, best_word))
    else:
        # Fallback
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
                overlap = 0
                max_o = min(len(new), len(block))
                for o in range(max_o, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
