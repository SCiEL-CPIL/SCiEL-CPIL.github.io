import time
import random
import itertools

# Precomputation for the specific case q=3, n=8
Q = 3
N = 8
R = 1
TOTAL_WORDS = Q ** N  # 6561

# Convert base-3 vector to int
def vec_to_int(vec):
    res = 0
    for v in vec:
        res = res * Q + v
    return res

# Convert int to base-3 vector
def int_to_vec(num, n=N, q=Q):
    vec = [0] * n
    for i in range(n - 1, -1, -1):
        vec[i] = num % q
        num //= q
    return vec

# Precompute balls for all words
# ball[i] is a list of ints j such that dist(i, j) <= R
BALLS = []
for i in range(TOTAL_WORDS):
    vec_i = int_to_vec(i)
    ball = [i]
    for pos in range(N):
        orig = vec_i[pos]
        for new_val in range(Q):
            if new_val == orig:
                continue
            vec_j = vec_i[:]
            vec_j[pos] = new_val
            j = vec_to_int(vec_j)
            ball.append(j)
    BALLS.append(ball)

# Note: dist(i,j) <= 1 is symmetric, so if j is in ball(i), i is in ball(j).
# We might want to precompute the "inverse" balls? 
# Actually, when we remove w from uncovered, we need to decrement counts[c] for all c such that w in ball(c).
# This is equivalent to c in ball(w). So we can just use BALLS[w].

def greedy_cover(seed):
    random.seed(seed)
    uncovered = set(range(TOTAL_WORDS))
    counts = [len(BALLS[i]) for i in range(TOTAL_WORDS)] # Initially 17 for all
    
    code = []
    
    while uncovered:
        # Find candidate with max count
        # To break ties randomly, we can pick randomly among maxes
        max_count = -1
        best_candidates = []
        # Optimization: just find max and one argmax, or list all?
        # Listing all maxes is O(N). 
        for i in range(TOTAL_WORDS):
            if counts[i] > max_count:
                max_count = counts[i]
                best_candidates = [i]
            elif counts[i] == max_count:
                best_candidates.append(i)
                
        if not best_candidates:
            break
            
        c_star = random.choice(best_candidates)
        code.append(c_star)
        
        # Update uncovered and counts
        # We iterate over the ball of c_star
        for w in BALLS[c_star]:
            if w in uncovered:
                uncovered.remove(w)
                # Decrement counts for all c that cover w
                # These are c in BALLS[w]
                for c in BALLS[w]:
                    counts[c] -= 1
                    
    return code

def construct(q, n, radius):
    start_time = time.time()
    deadline = 100.0  # 100s safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        # Simple random greedy or return empty/invalid
        # For safety, return a large set that covers everything? 
        # Or just the seed solution if it matches?
        # The task says "evaluator passes this task's alphabet size...". 
        # We assume it's always 3, 8, 1.
        # But to be robust:
        if n <= 4:
             # Brute force or small search
             pass
        # Just return a valid but potentially large code for other cases
        # Using the tetracode logic from seed if possible?
        # Let's just return a random set that is definitely covering (e.g. all words) 
        # or a heuristic.
        # Given the strict task description, we focus on 3,8,1.
        
        # Generic fallback: return all words? No, too big.
        # Return a random set? Might be invalid.
        # Let's stick to the specific case optimization.
        # If it's not the specific case, we can try the seed logic.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    best_code = None
    best_size = float('inf')
    
    seed_val = 42
    while True:
        current_time = time.time()
        if current_time - start_time > deadline:
            break
            
        code = greedy_cover(seed_val)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        # If we find a very good code, we might stop early? 
        # 486 is the target. If we get < 490, it's good.
        # But let's try to improve.
        
        seed_val += 1
        
    # Convert ints back to lists of ints
    if best_code is None:
        # Should not happen
        return []
        
    return [int_to_vec(c) for c in best_code]
