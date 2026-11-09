    def local_search(bitset, score, max_iters=500):
        """Perform local search by removing and re-adding elements."""
        nonlocal best_score, best_bitset
        for _ in range(max_iters):
            if time.time() - start_time > TIME_LIMIT:
                break
            # ... rest of the code
