<<<<<<< SEARCH
def _try_low_rank_search(n, m, p, deadline):
    """Search for low-rank decompositions starting from a reasonable lower bound."""
    for rank in range(14, 36):
        if time.time() > deadline:
            break

        num_restarts = 40 if rank <= 20 else 25
        result = _try_structured_rank(n, m, p, rank, deadline, num_restarts=num_restarts)
        if result is not None:
            return result

    return None
=======
def _try_low_rank_search(n, m, p, deadline):
    """Search for low-rank decompositions starting from a reasonable lower bound."""
    # Prioritize lower ranks with more restarts if time permits
    for rank in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 5.0: # Leave 5s for fallback
            break

        # Adjust restarts based on time remaining
        time_left = deadline - time.time()
        num_restarts = max(10, int(time_left / 2.0)) # ~2s per restart estimate
        
        result = _try_structured_rank(n, m, p, rank, deadline, num_restarts=num_restarts)
        if result is not None:
            return result

    return None
>>>>>>> REPLACE

