import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    def is_valid_triple(x, S):
        """Check if adding x to S creates a forbidden triple."""
        # Case 1: 1/x = 1/b + 1/c for distinct b,c in S
        # x = bc/(b+c) => bc = x(b+c) => bc - xb - xc = 0 => (b-x)(c-x) = x^2
        for b in S:
            if b == x:
                continue
            # (b-x)(c-x) = x^2, so c-x = x^2/(b-x) if b != x
            if b != x:
                num = x * x
                den = b - x
                if den != 0 and num % den == 0:
                    c = num // den + x
                    if c in S and c != x and c != b:
                        return True
        # Case 2: 1/b = 1/x + 1/c for distinct x,c in S, b in S
        # b = xc/(x+c) => xc = b(x+c) => xc - bx - bc = 0 => (b-x)(b-c) = b^2 ... 
        # Actually: 1/b = 1/x + 1/c => c = bx/(b-x) for b > x
        for b in S:
            if b == x:
                continue
            if b > x:
                num = b * x
                den = b - x
                if den != 0 and num % den == 0:
                    c = num // den
                    if c in S and c != x and c != b:
                        return True
        # Case 3: 1/c = 1/x + 1/b for distinct x,b in S, c in S
        # c = xb/(x+b) => same as case 2 with roles swapped
        for c in S:
            if c == x:
                continue
            if c > x:
                num = c * x
                den = c - x
                if den != 0 and num % den == 0:
                    b = num // den
                    if b in S and b != x and b != c:
                        return True
        return False
    
    # Try different orderings to maximize set size
    best_set = set()
    
    # Strategy 1: Add numbers in descending order (larger numbers are less likely to conflict)
    S = set()
    for x in range(180, 0, -1):
        if time.time() > deadline:
            break
        if not is_valid_triple(x, S):
            S.add(x)
    
    if len(S) > len(best_set):
        best_set = S.copy()
    
    # Strategy 2: Add numbers in ascending order
    S2 = set()
    for x in range(1, 181):
        if time.time() > deadline:
            break
        if not is_valid_triple(x, S2):
            S2.add(x)
    
    if len(S2) > len(best_set):
        best_set = S2.copy()
    
    # Strategy 3: Start with large numbers, then try to add smaller ones
    S3 = set(range(91, 181))  # Start with upper half
    # Verify this is valid
    valid = True
    S3_list = list(S3)
    for i in range(len(S3_list)):
        for j in range(i+1, len(S3_list)):
            a, b = S3_list[i], S3_list[j]
            # Check if there exists c in S3 such that 1/a = 1/b + 1/c
            # c = ab/(b-a) if b > a
            if b > a:
                num = a * b
                den = b - a
                if num % den == 0:
                    c = num // den
                    if c in S3 and c != a and c != b:
                        valid = False
                        break
        if not valid:
            break
    
    if valid:
        if len(S3) > len(best_set):
            best_set = S3.copy()
    
    # Try to extend S3 by adding smaller numbers
    for x in range(90, 0, -1):
        if time.time() > deadline:
            break
        if not is_valid_triple(x, S3):
            S3.add(x)
    
    if len(S3) > len(best_set):
        best_set = S3.copy()
    
    # Strategy 4: Greedy with random-like ordering (use a deterministic shuffle)
    import random
    rng = random.Random(42)
    order = list(range(1, 181))
    rng.shuffle(order)
    S4 = set()
    for x in order:
        if time.time() > deadline:
            break
        if not is_valid_triple(x, S4):
            S4.add(x)
    
    if len(S4) > len(best_set):
        best_set = S4.copy()
    
    # Strategy 5: Try to maximize by being more selective
    # Start empty, add in order of least conflicts
    S5 = set()
    candidates = list(range(1, 181))
    while candidates and time.time() < deadline:
        best_x = None
        min_conflicts = float('inf')
        for x in candidates:
            # Count how many pairs in S would conflict with x
            conflicts = 0
            for b in S5:
                if b == x:
                    continue
                # Check all three cases
                if b > x:
                    num = b * x
                    den = b - x
                    if den != 0 and num % den == 0:
                        c = num // den
                        if c in S5 and c != x and c != b:
                            conflicts += 1
                # 1/x = 1/b + 1/c case
                if b != x:
                    num = x * x
                    den = b - x
                    if den != 0 and num % den == 0:
                        c = num // den + x
                        if c in S5 and c != x and c != b:
                            conflicts += 1
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_x = x
                if conflicts == 0:
                    break
        if best_x is not None:
            S5.add(best_x)
            candidates.remove(best_x)
    
    if len(S5) > len(best_set):
        best_set = S5.copy()
    
    return sorted(best_set)
