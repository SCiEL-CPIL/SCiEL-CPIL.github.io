import math
import time

def calculate_score(U):
    if not U:
        return 0.0
    U_set = set(U)
    max_u = max(U_set)
    
    # Calculate sumset
    sumset = set()
    diffset = set()
    n = len(U_set)
    # Using list for iteration
    U_list = list(U_set)
    
    for i in range(n):
        for j in range(i, n):
            sumset.add(U_list[i] + U_list[j])
        for j in range(n):
            diffset.add(U_list[i] - U_list[j])
            
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    # log base 2 is fine as it's a constant factor in the fraction, 
    # actually the formula uses log(2 max U + 1) in denominator.
    # The base of the log in the numerator and denominator cancels out if they are the same?
    # Formula: 1 + log(|U-U|/|U+U|) / log(2 max U + 1)
    # If both are log base e, it's fine. If both are log base 2, it's fine.
    # The prompt says "log", usually natural log in math, but ratio is independent of base 
    # as long as it's consistent.
    
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0 # max_u = 0, U={0}, ratio 1/1=1, log(1)=0, score 1
        
    score = 1 + math.log(ratio) / denom
    return score

def is_sidon(U):
    """Check if U is a Sidon set (distinct sums for i<=j)."""
    U_list = sorted(U)
    n = len(U_list)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = U_list[i] + U_list[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def generate_minimal_sidon(n):
    """
    Generate a Sidon set of size n with the smallest possible maximum element
    using a greedy approach.
    """
    if n == 1:
        return [0]
    
    U = [0]
    # We will try to add elements one by one
    # To ensure "minimal max", we should try to add the smallest possible next element
    # that keeps the set Sidon.
    
    # However, greedy "smallest next" might not yield the globally minimal max for size n.
    # But it's a good heuristic and fast.
    # A better heuristic: Backtracking? 
    # For n up to 30, greedy is likely sufficient to find good candidates.
    # Let's stick to greedy for speed, but maybe try a few variations?
    # Actually, the "greedy" Sidon set is well-studied.
    
    current_max = 0
    for _ in range(n - 1):
        candidate = current_max + 1
        while True:
            new_U = U + [candidate]
            if is_sidon(new_U):
                U = new_U
                current_max = candidate
                break
            candidate += 1
            
    return U

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_score = -1
    best_set = [0, 1, 3]  # Seed
    
    # Evaluate seed
    seed_score = calculate_score(best_set)
    best_score = seed_score
    
    # Generate Sidon sets of increasing size
    # We go up to size 100, but check time
    for n in range(3, 101):
        if time.time() > deadline:
            break
            
        try:
            U = generate_minimal_sidon(n)
        except Exception:
            continue
            
        score = calculate_score(U)
        
        if score > best_score:
            best_score = score
            best_set = U

    # Optional: If time remains, try to optimize the best set found 
    # by swapping elements? 
    # Given the structure, Sidon sets are likely optimal locally.
    # We'll just return the best found.
    
    return best_set
