    for n_sizes, m_sizes, p_sizes in candidates:
        if time.time() > deadline - 5.0:
            break
        if sum(n_sizes) != n or sum(m_sizes) != m or sum(p_sizes) != p:
            continue
        try:
            U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
            # Add verification
            if verify_factors(U, V, W, n, m, p):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_U, best_V, best_W = U, V, W
        except Exception:
            continue
