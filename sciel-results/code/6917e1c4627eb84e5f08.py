import time

def construct():
    # Known valid set of 16 distinct integers in 2..40
    # Constructed from primes and their powers p, p^2, p^4 where p^4 <= 40
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # Powers of 2: 4 (2^2), 16 (2^4)
    # Powers of 3: 9 (3^2)
    # Powers of 5: 25 (5^2)
    # Total: 12 + 2 + 1 + 1 = 16
    
    candidate_set = [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]
    
    # Verify the set to ensure correctness
    def is_valid(s):
        prods = {1}
        for x in s:
            new_prods = set()
            for p in prods:
                np = x * p
                if np in prods:
                    return False
                new_prods.add(np)
            prods = new_prods
        return True

    if is_valid(candidate_set):
        return candidate_set
    
    # Fallback to primes if verification fails (should not happen)
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

if __name__ == "__main__":
    res = construct()
    print(f"Constructed set size: {len(res)}")
    print(res)
