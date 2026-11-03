def rank_of(n_s, m_s, p_s):
    """Compute rank for given splits without building matrices."""
    total = 0
    for n_b in n_s:
        for m_b in m_s:
            for p_b in p_s:
                if n_b == 2 and m_b == 2 and p_b == 2:
                    total += 7
                else:
                    total += n_b * m_b * p_b
    return total
