    def search_rank(R, num_restarts=20, deadline_time=None):
        """Search for a rank-R decomposition"""
        for restart in range(num_restarts):
            if deadline_time is not None and time.time() > deadline_time:
                return None
            # ... rest of the code
