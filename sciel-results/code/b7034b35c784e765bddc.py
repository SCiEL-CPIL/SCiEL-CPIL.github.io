import itertools

def construct():
    # Generate Hamming numbers (5-smooth) up to 10000
    # Actually 2,3-smooth is enough for the current set, 
    # but 5-smooth gives more options for collisions?
    # Let's stick to 2,3-smooth (Hamming) as in current, 
    # but try to find a better subset.
    
    # Generate all 2^i * 3^j <= 10000
    hammers = set()
    for i in range(15): # 2^14 = 16384 > 10000
        for j in range(10): # 3^9 = 19683 > 10000
            val = (2 ** i) * (3 ** j)
            if val <= 10000:
                hammers.add(val)
    hammers = sorted(list(hammers))
    
    # The current set is the first 20.
    # Let's try to find a subset of 20 that minimizes max(|A+A|, |A*A|).
    # This is a combinatorial optimization problem.
    # We can try sliding windows or random sampling.
    
    best_set = hammers[:20]
    best_val = float('inf')
    
    # Helper to compute score
    def calc_score(A):
        sums = set()
        prods = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
    
    current_score = calc_score(best_set)
    best_val = current_score
    
    # Try sliding windows of size 20 in the sorted Hamming list
    for start in range(len(hammers) - 19):
        candidate = hammers[start:start+20]
        score = calc_score(candidate)
        if score < best_val:
            best_val = score
            best_set = candidate
            
    # Try some random subsets?
    # Given time constraints, sliding window might be sufficient.
    # Also try mixing small and large?
    
    # Let's also try the set of 20 smallest 5-smooth numbers (Hamming)
    # 5-smooth numbers: 2^i 3^j 5^k
    hammers5 = set()
    for i in range(15):
        for j in range(10):
            for k in range(8):
                val = (2 ** i) * (3 ** j) * (5 ** k)
                if val <= 10000:
                    hammers5.add(val)
    hammers5 = sorted(list(hammers5))
    
    # Check first 20 5-smooth
    cand5 = hammers5[:20]
    score5 = calc_score(cand5)
    if score5 < best_val:
        best_val = score5
        best_set = cand5
        
    # Check sliding windows in 5-smooth
    for start in range(len(hammers5) - 19):
        candidate = hammers5[start:start+20]
        score = calc_score(candidate)
        if score < best_val:
            best_val = score
            best_set = candidate

    return best_set
