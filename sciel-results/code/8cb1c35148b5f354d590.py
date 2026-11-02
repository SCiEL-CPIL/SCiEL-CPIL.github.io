        # 2. Build 4x1x6 Rank-1 (Rank 1)
        U2, V2, W2 = _build_rank1_4x1x6()
        decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
