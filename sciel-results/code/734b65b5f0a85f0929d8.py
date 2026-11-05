    def prune_optimized(S):
        """Try to remove elements while maintaining maximality, using cached sums."""
        S = list(S)
        sums = get_sums_bitset(S)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return S
            indices = list(range(len(S)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                new_sums = get_sums_bitset(new_S)
                if is_maximal_fast(new_S, new_sums):
                    S = new_S
                    sums = new_sums
                    improved = True
                    break
        return S
