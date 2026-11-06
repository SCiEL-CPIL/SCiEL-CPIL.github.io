    def can_add(S_mask, counts, bad_sums, x):
        if counts[2 * x] >= 2:
            return False
        if np.any(S_mask[1:N+1] & bad_sums[x+1:x+N+1]):
            return False
        return True
