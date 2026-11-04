<<<<<<< SEARCH
    def try_rank(target_rank):
        """
        Try both ALS and L-BFGS-B for the given rank.
        """
        # Try ALS first (often faster for structured problems)
        if try_rank_als(target_rank, n_restarts=3, max_iter=30):
            return True
        # Then try L-BFGS-B
        if try_rank_l_bfgs(target_rank, n_restarts=2):
            return True
        return False
=======
    def try_rank(target_rank):
        """
        Try both ALS and L-BFGS-B for the given rank.
        """
        # Try ALS first (often faster for structured problems)
        if try_rank_als(target_rank, n_restarts=5, max_iter=50):
            return True
        # Then try L-BFGS-B
        if try_rank_l_bfgs(target_rank, n_restarts=3):
            return True
        return False
>>>>>>> REPLACE

