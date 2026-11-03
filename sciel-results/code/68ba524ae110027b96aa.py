import itertools
import random
import time

def construct(q, n, radius):
    """
    Construct a covering code using greedy set cover with randomized restarts and local search.
    """
    if n == 0:
        return [[]]
    
    # For this specific task, q=3, n=6, radius=1
    # But we should handle general cases if possible, or at least not crash.
    # The problem statement says "evaluator passes this task's alphabet size q, length n and covering radius".
    # So we assume q=3, n=6, radius=1 based on the context, but the function signature takes them.
    
    # Precomputation for speed
    # Map words to indices
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    N = len(words)
    
    # Precompute balls
    # ball[i] is a list of indices of words within Hamming distance 'radius' of words[i]
    # For radius=1, ball size is 1 + n*(q-1)
    balls = []
    for i, w in enumerate(words):
        ball = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                ball.append(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    best_code = None
    best_size = float('inf')
    
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Helper function to check if a set of codewords covers the space
    # and to perform local search reduction
    def local_search_reduction(code, covered_mask):
        """
        code: list of indices
        covered_mask: not passed, we recompute or maintain. 
        For efficiency in reduction, we maintain a count of how many times each word is covered.
        """
        # Count coverage for each word
        # cover_count[j] = number of codewords c in code such that j in balls[c]
        cover_count = [0] * N
        for c in code:
            for j in balls[c]:
                cover_count[j] += 1
        
        # Check if all covered
        for j in range(N):
            if cover_count[j] == 0:
                return code, cover_count # Not a valid code, or we shouldn't have entered here if we maintained validity
                
        # Try to remove codewords
        new_code = list(code)
        removed = False
        # Repeat until no more removals possible
        while True:
            removed_any = False
            # Iterate backwards to allow removal without index shifting issues
            for i in range(len(new_code) - 1, -1, -1):
                c = new_code[i]
                # Check if c is redundant
                redundant = True
                for j in balls[c]:
                    if cover_count[j] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    for j in balls[c]:
                        cover_count[j] -= 1
                    new_code.pop(i)
                    removed_any = True
            if not removed_any:
                break
                
        return new_code, cover_count

    # Greedy construction
    def greedy_construct(randomize=False):
        covered = [False] * N
        code = []
        
        # To speed up greedy, we can maintain a list of uncovered words
        # and for each candidate, count how many it covers.
        # However, simple iteration is fast enough for N=729.
        
        uncovered_count = N
        while uncovered_count > 0:
            best_c = -1
            best_gain = -1
            best_gain_ties = []
            
            for c in range(N):
                gain = 0
                for j in balls[c]:
                    if not covered[j]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    best_gain_ties = [c]
                elif gain == best_gain and gain > 0:
                    best_gain_ties.append(c)
            
            if best_gain == 0:
                # Should not happen if uncovered_count > 0
                break
                
            if randomize and len(best_gain_ties) > 1:
                best_c = random.choice(best_gain_ties)
            
            code.append(best_c)
            for j in balls[best_c]:
                if not covered[j]:
                    covered[j] = True
                    uncovered_count -= 1
        
        return code

    # Initial greedy
    code = greedy_construct(randomize=False)
    # Reduce
    code, _ = local_search_reduction(code, None)
    best_code = code
    best_size = len(code)
    
    # Randomized restarts
    while time.time() - start_time < time_limit:
        code = greedy_construct(randomize=True)
        code, _ = local_search_reduction(code, None)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            # If we reach the theoretical lower bound or known best, we can stop early?
            # Lower bound is 71. Best known 73.
            if best_size <= 73:
                # We can try to polish further, but 73 is likely the target.
                # Let's continue a bit to see if 72 or 71 is possible, but 73 is a strong target.
                pass

    # Convert indices back to words
    return [words[i] for i in best_code]
