    def find_valid_residue(m_idx):
        m = moduli[m_idx]
        start = random.randint(0, m - 1)
        for k in range(m):
            r = (start + k) % m
            if is_valid(m_idx, r):
                return r
        return None
