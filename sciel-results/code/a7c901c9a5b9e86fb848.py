import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0
    
    total_words = q ** n
    
    # Precompute balls
    # For efficiency, we assume radius=1 for the fast path.
    # If radius != 1, we generate balls generally.
    balls_list = []
    ball_size = 0
    
    if radius == 1:
        # Fast path for radius 1
        # Neighbors: self + n * (q-1) changes
        ball_size = 1 + n * (q - 1)
        balls_arr = np.zeros((total_words, ball_size), dtype=np.int32)
        
        for i in range(total_words):
            digits = [0] * n
            temp = i
            for j in range(n):
                digits[j] = temp % q
                temp //= q
            
            neighbors = [i]
            for j in range(n):
                orig = digits[j]
                for val in range(q):
                    if val == orig: continue
                    new_digit = digits.copy()
                    new_digit[j] = val
                    new_idx = 0
                    power = 1
                    for k in range(n):
                        new_idx += new_digit[k] * power
                        power *= q
                    neighbors.append(new_idx)
            balls_arr[i] = neighbors
    else:
        # General path (slower)
        # Generate all words within Hamming distance <= radius
        # This can be expensive for large n, q, R.
        # But for this task, it's likely R=1.
        # We'll just generate all words and filter? No, too slow.
        # We'll generate neighbors by changing positions.
        # For simplicity and safety, if R!=1, we might just return a trivial code
        # or use a slower generation.
        # Given the task is K_3(7,1), I will assume R=1.
        # If R != 1, I will fall back to a simple greedy without precomputed balls?
        # No, let's just handle R=1 efficiently.
        # If R != 1, I will raise an error or use a very basic method.
        # But the prompt says "this task's ... radius".
        # I'll stick to R=1 optimization.
        
        # Fallback for R != 1: Just return all words? 
        # No, let's try to generate balls for small R.
        # For R=1, it's 15.
        # For R=2, it's 1 + 14 + 21*4 = 1 + 14 + 84 = 99.
        # Still manageable.
        
        # Let's write a general ball generator for small R.
        def gen_balls_general(q, n, R):
            balls = []
            for i in range(total_words):
                digits = [0] * n
                temp = i
                for j in range(n):
                    digits[j] = temp % q
                    temp //= q
                
                neighbors = set([i])
                # Generate all words within distance R
                # This is complex to do efficiently without recursion.
                # For R=1, we did it above.
                # For R>1, we can do BFS or just iterate all words and check distance?
                # 2187 * 2187 = 4.7M distance checks.
                # Distance check is sum of abs diff? No, Hamming.
                # 4.7M * 7 ops = 33M ops.
                # This is doable in NumPy if vectorized, or even Python if optimized.
                # But let's assume R=1.
                
                # If R != 1, I will use the "check all words" method for ball generation
                # which is O(q^n * q^n * n).
                # For 3^7, this is fine.
                
                all_words_arr = np.array([list(map(int, format(j, '0' + str(n) + 'b').zfill(n))) for j in range(total_words)]) # This is base 2 logic, wrong for q=3.
                
                # Correct general generation:
                # Create array of all words
                words_arr = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
                
                # For each i, find words with Hamming distance <= R
                # dist = sum(words_arr != digits, axis=1)
                # mask = dist <= R
                # balls.append(np.where(mask)[0])
                
                # This is O(total_words * total_words * n).
                # 2187 * 2187 * 7 ~ 33M ops.
                # In Python loop, this is slow.
                # In NumPy, `words_arr != digits` creates a (2187, 7) array.
                # `sum(axis=1)` is fast.
                # Doing this 2187 times is 2187 * (2187*7) ops ~ 33M.
                # This might take a few seconds.
                
                # I will implement this general case.
                
            return balls

        # Actually, let's just use the R=1 path. If R!=1, I'll use a simple fallback.
        # The task is specific.
        
        # To be safe, I'll implement the general ball generation using NumPy vectorization per row.
        words_arr = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        for i in range(total_words):
            digits = words_arr[i]
            dists = np.sum(words_arr != digits, axis=1)
            mask = dists <= radius
            balls_list.append(np.where(mask)[0])
            if i == 0:
                ball_size = len(balls_list[0])
        
        # Convert to array
        balls_arr = np.zeros((total_words, ball_size), dtype=np.int32)
        for i in range(total_words):
            balls_arr[i] = balls_list[i]

    best_code_indices = None
    best_size = total_words + 1
    
    trials = 0
    while time.time() < deadline:
        trials += 1
        code_indices, size = _greedy_reduce(total_words, balls_arr, trials)
        if size < best_size:
            best_size = size
            best_code_indices = code_indices
        if best_size <= 186:
            break
            
    if best_code_indices is None:
        # Fallback
        return [list(t) for t in itertools.product(range(q), repeat=n)]
        
    # Convert indices to words
    result = []
    for idx in best_code_indices:
        word = [0] * n
        temp = idx
        for j in range(n):
            word[j] = temp % q
            temp //= q
        result.append(word)
        
    return result

def _greedy_reduce(total_words, balls_arr, random_state):
    rng = np.random.default_rng(random_state)
    coverage = np.zeros(total_words, dtype=np.int32)
    code = []
    
    # Greedy Phase
    while True:
        if not np.any(coverage == 0):
            break
            
        gains = np.sum(coverage[balls_arr] == 0, axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        candidates = np.where(gains == max_gain)[0]
        idx = rng.choice(candidates)
        
        code.append(idx)
        coverage[balls_arr[idx]] += 1
        
    # Reduction Phase
    changed = True
    while changed:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c_idx = code[i]
            if np.min(coverage[balls_arr[c_idx]]) >= 2:
                coverage[balls_arr[c_idx]] -= 1
                code.pop(i)
                changed = True
                
    return code, len(code)
