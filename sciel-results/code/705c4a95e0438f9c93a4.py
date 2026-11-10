import numpy as np
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free set modulo 205.
    
    Key insight: By CRT, 205 = 5 * 41. A number d mod 205 is a nonzero square
    iff d mod 5 is a nonzero square mod 5 AND d mod 41 is a nonzero square mod 41.
    
    Nonzero squares mod 5: {1, 4}. Non-squares mod 5: {2, 3}.
    
    If we pick two residues a, b mod 5 such that (a-b) mod 5 is a non-square mod 5
    (i.e., in {2, 3}), then for ANY k1, k2 mod 41, the difference
    (a + 5*k1) - (b + 5*k2) mod 205 has mod 5 part (a-b) mod 5 which is a non-square,
    so the difference is NEVER a square mod 205.
    
    Therefore, we can take ALL 41 residues mod 41 lifted to two different residue
    classes mod 5 that differ by a non-square mod 5. This gives 41 + 41 = 82 elements.
    
    Using a=0, b=2 (since 2 is a non-square mod 5):
      - All elements 5*k mod 205 for k = 0..40
      - All elements (2 + 5*k) mod 205 for k = 0..40
    """
    m = 205
    
    # Construct the set: residues 0 mod 5 and 2 mod 5, all residues mod 41
    result = set()
    for k in range(41):
        result.add(5 * k % m)          # residue 0 mod 5
        result.add((2 + 5 * k) % m)    # residue 2 mod 5
    
    result_arr = np.array(sorted(result), dtype=np.int64)
    
    # Verify validity
    forbidden_205 = set()
    for x in range(1, m):
        forbidden_205.add(pow(x, 2, m))
    forbidden_205.discard(0)
    
    valid = True
    for i in range(len(result_arr)):
        for j in range(i+1, len(result_arr)):
            d1 = (result_arr[i] - result_arr[j]) % m
            d2 = (result_arr[j] - result_arr[i]) % m
            if d1 in forbidden_205 or d2 in forbidden_205:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        print("VALIDATION FAILED, using trivial set", file=sys.stderr)
        return np.array([0], dtype=np.int64)
    
    print(f"Final result size: {len(result_arr)}", file=sys.stderr)
    return result_arr
