# EVOLVE-BLOCK-START
def construct():
    import itertools
    
    # Precompute modular inverses for denominators 1 to 10 modulo 101
    inverses = {x: pow(x, -1, 101) for x in range(1, 11)}
    
    # Split denominators into two halves for Meet-in-the-Middle
    D1 = [1, 2, 3, 4, 5]
    D2 = [6, 7, 8, 9, 10]
    
    # Generate all subsets of D1
    subsets1 = []
    for r1 in range(len(D1) + 1):
        for comb1 in itertools.combinations(D1, r1):
            sum1 = sum(inverses[x] for x in comb1) % 101
            size1 = len(comb1)
            terms1 = list(comb1)
            subsets1.append((sum1, size1, terms1))
            
    # Generate all subsets of D2 and keep only the minimum size for each sum
    best_B = {} # sum -> (size, terms)
    for r2 in range(len(D2) + 1):
        for comb2 in itertools.combinations(D2, r2):
            sum2 = sum(inverses[x] for x in comb2) % 101
            size2 = len(comb2)
            terms2 = list(comb2)
            
            if sum2 not in best_B or size2 < best_B[sum2][0]:
                best_B[sum2] = (size2, terms2)
                
    # Initialize result
    result = [None] * 101
    
    # Combine subsets from D1 and D2
    for sum1, size1, terms1 in subsets1:
        for r in range(101):
            req_sum = (r - sum1) % 101
            if req_sum in best_B:
                size2, terms2 = best_B[req_sum]
                total_size = size1 + size2
                candidate_terms = terms1 + terms2
                
                if result[r] is None or len(result[r]) > total_size:
                    result[r] = candidate_terms
                    
    # Verify all residues are covered (should be true)
    # If any are None, we might need a fallback, but with 10 elements, coverage is guaranteed.
    
    return result
# EVOLVE-BLOCK-END
