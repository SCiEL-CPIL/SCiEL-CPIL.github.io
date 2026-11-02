<<<<<<< SEARCH
    # Phase 2: Local search on best solution
    if best_blk_idxs is not None:
        # Remove redundant first
        best_blk_idxs = remove_redundant(best_blk_idxs, deadline - 10.0)
        best_len = len(best_blk_idxs)

        # Iterative local search
        for iteration in range(50):
            if time.time() > deadline - 10.0:
                break

            # Single swap
            improved = False
            prev_len = len(best_blk_idxs)
            best_blk_idxs = local_search_single_swap(
                best_blk_idxs, max_passes=80, time_limit=deadline - 10.0
            )
            if len(best_blk_idxs) < prev_len:
                improved = True
                best_len = len(best_blk_idxs)

            # Two exchange
            if time.time() > deadline - 10.0:
                break
            prev_len = len(best_blk_idxs)
            best_blk_idxs = local_search_two_exchange(
                best_blk_idxs, max_passes=40, time_limit=deadline - 10.0
            )
            if len(best_blk_idxs) < prev_len:
                improved = True
                best_len = len(best_blk_idxs)

            if not improved:
                # Try more aggressive single swap
                prev_len = len(best_blk_idxs)
                best_blk_idxs = local_search_single_swap(
                    best_blk_idxs, max_passes=200, time_limit=deadline - 8.0
                )
                if len(best_blk_idxs) < prev_len:
                    best_len = len(best_blk_idxs)
                else:
                    break

        # Final polish
        if time.time() < deadline - 5.0:
            for _ in range(20):
                if time.time() > deadline - 4.0:
                    break
                prev_len = len(best_blk_idxs)
                best_blk_idxs = local_search_single_swap(
                    best_blk_idxs, max_passes=150, time_limit=deadline - 4.0
                )
                if len(best_blk_idxs) < prev_len:
                    best_len = len(best_blk_idxs)
                else:
                    break
=======
    # Phase 2: Local search on best solution
    if best_blk_idxs is not None:
        # Remove redundant first
        best_blk_idxs = remove_redundant(best_blk_idxs, deadline - 10.0)
        best_len = len(best_blk_idxs)

        # Iterative local search with increasing effort
        for iteration in range(80):
            if time.time() > deadline - 10.0:
                break

            # Single swap with increasing passes
            passes = 60 + iteration * 3
            improved = False
            prev_len = len(best_blk_idxs)
            best_blk_idxs = local_search_single_swap(
                best_blk_idxs, max_passes=passes, time_limit=deadline - 10.0
            )
            if len(best_blk_idxs) < prev_len:
                improved = True
                best_len = len(best_blk_idxs)

            # Two exchange
            if time.time() > deadline - 10.0:
                break
            prev_len = len(best_blk_idxs)
            best_blk_idxs = local_search_two_exchange(
                best_blk_idxs, max_passes=30 + iteration * 2, time_limit=deadline - 10.0
            )
            if len(best_blk_idxs) < prev_len:
                improved = True
                best_len = len(best_blk_idxs)

            if not improved:
                # Try more aggressive single swap with longer time
                prev_len = len(best_blk_idxs)
                best_blk_idxs = local_search_single_swap(
                    best_blk_idxs, max_passes=300, time_limit=deadline - 8.0
                )
                if len(best_blk_idxs) < prev_len:
                    best_len = len(best_blk_idxs)
                else:
                    # One more round of two-exchange before giving up
                    if time.time() > deadline - 6.0:
                        break
                    prev_len = len(best_blk_idxs)
                    best_blk_idxs = local_search_two_exchange(
                        best_blk_idxs, max_passes=100, time_limit=deadline - 6.0
                    )
                    if len(best_blk_idxs) < prev_len:
                        best_len = len(best_blk_idxs)
                    else:
                        break

        # Final polish with high precision
        if time.time() < deadline - 5.0:
            for _ in range(30):
                if time.time() > deadline - 4.0:
                    break
                prev_len = len(best_blk_idxs)
                best_blk_idxs = local_search_single_swap(
                    best_blk_idxs, max_passes=200, time_limit=deadline - 4.0
                )
                if len(best_blk_idxs) < prev_len:
                    best_len = len(best_blk_idxs)
                else:
                    break
            # Final two-exchange polish
            if time.time() < deadline - 3.0:
                for _ in range(10):
                    if time.time() > deadline - 3.0:
                        break
                    prev_len = len(best_blk_idxs)
                    best_blk_idxs = local_search_two_exchange(
                        best_blk_idxs, max_passes=100, time_limit=deadline - 3.0
                    )
                    if len(best_blk_idxs) < prev_len:
                        best_len = len(best_blk_idxs)
                    else:
                        break
>>>>>>> REPLACE

