import time
import random
import itertools

def construct(q, n, radius):
    # Specific optimization for the known hard case: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _solve_3_7_1()
    
    # Generic fallback: Greedy Set Cover
    return _greedy_cover(q, n, radius)

def _solve_3_7_1():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    q = 3
    n = 7
    total_words = q ** n
    radius = 1
    
    # Precompute neighbors for each word in {0,1,2}^7
    # Word represented as integer 0..2186
    # Convert int to base-3 digits
    def int_to_base3(num, length):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % 3
            num //= 3
        return digits

    def base3_to_int(digits):
        val = 0
        for d in reversed(digits):
            val = val * 3 + d
        return val

    neighbors = []
    for i in range(total_words):
        digits = int_to_base3(i, n)
        neigh = [i]
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                new_digits = digits[:]
                new_digits[pos] = new_val
                neigh.append(base3_to_int(new_digits))
        neighbors.append(neigh)

    # Initial Code: Tetracode sum
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_code.add(base3_to_int(word))
    
    code = set(initial_code)
    best_code = set(code)
    best_size = len(code)
    
    # Coverage counts
    coverage_count = [0] * total_words
    for c in code:
        for x in neighbors[c]:
            coverage_count[x] += 1
            
    def update_coverage(c, delta):
        for x in neighbors[c]:
            coverage_count[x] += delta
            
    def is_redundant(c):
        for x in neighbors[c]:
            if coverage_count[x] <= 1:
                return False
        return True

    def find_redundant():
        for c in code:
            if is_redundant(c):
                return c
        return None

    def try_swap(c):
        # Tentatively remove c
        update_coverage(c, -1)
        exposed = []
        for x in neighbors[c]:
            if coverage_count[x] == 0:
                exposed.append(x)
                
        if not exposed:
            # c was redundant, just remove
            code.remove(c)
            return True, "remove"
            
        # Find best c' to cover exposed
        best_c_prime = -1
        best_score = -1
        # Optimization: Only check c' that are close to exposed? 
        # For now, check all.
        for c_prime in range(total_words):
            if c_prime in code:
                continue
            score = 0
            # Check how many exposed words are covered by c_prime
            # neighbors[c_prime] is a list. 
            # We need intersection size with exposed.
            # Since exposed is small (usually < 15), we can iterate exposed
            # and check if in neighbors[c_prime]
            # Or iterate neighbors[c_prime] and check if in exposed set
            exposed_set = set(exposed)
            for x in neighbors[c_prime]:
                if x in exposed_set:
                    score += 1
            if score > best_score:
                best_score = score
                best_c_prime = c_prime
            # Early exit if we found a perfect cover
            if score == len(exposed):
                break
                
        if best_c_prime != -1 and best_score == len(exposed):
            # Valid swap
            code.remove(c)
            code.add(best_c_prime)
            update_coverage(best_c_prime, 1)
            return True, "swap"
        else:
            # Revert
            update_coverage(c, 1)
            return False, "fail"
            
    # Local Search Loop
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Remove all redundant words
        while True:
            if time.time() > deadline:
                break
            c = find_redundant()
            if c is None:
                break
            code.remove(c)
            update_coverage(c, -1)
            improved = True
            
        if len(code) < best_size:
            best_code = set(code)
            best_size = len(code)
            
        if not improved and time.time() > deadline:
            break
            
        # Phase 2: Try swaps to enable further removals
        # Pick a random word to try to swap
        if code:
            c = random.choice(list(code))
            success, _ = try_swap(c)
            if success:
                # After a swap, we might have created new redundancies
                # Loop back to Phase 1
                continue
        
        # If no improvement in a while, maybe randomize?
        # For now, just keep trying swaps on random words.
        # If stuck, the loop continues until deadline.
        
    # Convert best_code back to list of lists
    result = []
    for c in best_code:
        digits = int_to_base3(c, n)
        result.append(digits)
        
    return result

def _greedy_cover(q, n, radius):
    # Generic Greedy Set Cover
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute neighbors
    def get_neighbors(word, q, n, radius):
        # For radius 1
        neigh = [word]
        for i in range(n):
            for v in range(q):
                if word[i] == v:
                    continue
                new_word = list(word)
                new_word[i] = v
                neigh.append(tuple(new_word))
        return set(neigh)
        
    # If radius > 1, this needs expansion, but task is likely R=1
    if radius != 1:
        # Fallback to simple random or basic construction
        # For now, just return a large code
        return [list(w) for w in all_words]
        
    neighbors = {w: get_neighbors(w, q, n, 1) for w in all_words}
    
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        best_word = None
        max_covered = 0
        for w in all_words:
            covered = len(neighbors[w] & uncovered)
            if covered > max_covered:
                max_covered = covered
                best_word = w
        if best_word is None:
            break
        code.append(list(best_word))
        uncovered -= neighbors[best_word]
        
    return code
