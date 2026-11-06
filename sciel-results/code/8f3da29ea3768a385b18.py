import itertools

def construct():
    # Modular inverses of 1..10 mod 101
    invs = [pow(i, -1, 101) for i in range(1, 11)]
    
    def try_max_size(K):
        # Map residue to the first subset (smallest size, then smallest mask order) found
        # We iterate by size, so the first one found for a residue is the smallest size.
        # Among same size, combinations are generated in lexicographic order of indices.
        res_map = {r: None for r in range(101)}
        
        for size in range(1, K + 1):
            for comb in itertools.combinations(range(10), size):
                s = sum(invs[i] for i in comb) % 101
                if res_map[s] is None:
                    res_map[s] = comb
        
        # Check if all residues are covered
        for r in range(101):
            if res_map[r] is None:
                return None
        
        return res_map

    # Try to find a solution with max subset size 3, then 4, etc.
    solution = None
    for K in range(3, 11):
        solution = try_max_size(K)
        if solution is not None:
            break
            
    if solution is None:
        # Fallback: use the seed logic just in case (though impossible for K=10)
        representations = {}
        for mask in range(1, 1 << 10):
            terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
            residue = sum(invs[x - 1] for x in terms) % 101
            if residue not in representations:
                representations[residue] = terms
        # This might miss some if not all masks checked? No, 1023 masks cover all subsets.
        # But we need to ensure all 101 are present.
        # If K=10 failed, it means some residue has no subset? Impossible.
        # So solution should never be None.
        pass

    result = []
    for r in range(101):
        if solution[r] is None:
            # Should not happen
            result.append([1])
        else:
            result.append([i + 1 for i in solution[r]])
            
    return result
