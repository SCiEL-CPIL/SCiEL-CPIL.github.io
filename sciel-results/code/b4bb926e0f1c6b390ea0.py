<<<<<<< SEARCH
    # Second pair replacement pass after final reduction
    if time.time() < deadline - 10.0:
        selected, cover_count = pair_replace(selected, cover_count, max_trials=200)
        if selected:
            cc = np.zeros(n_t, dtype=np.int32)
            for bi in selected:
                cc[block_covers_np[bi]] += 1
            selected, cover_count = reduce_blocks(selected, cc)
=======
    # Second pair replacement pass after final reduction
    if time.time() < deadline - 8.0:
        selected, cover_count = pair_replace(selected, cover_count, max_trials=500)
        if selected:
            cc = np.zeros(n_t, dtype=np.int32)
            for bi in selected:
                cc[block_covers_np[bi]] += 1
            selected, cover_count = reduce_blocks(selected, cc)

    # Final aggressive reduction pass
    if time.time() < deadline - 5.0:
        selected, cover_count = reduce_blocks(selected, cover_count)
>>>>>>> REPLACE

