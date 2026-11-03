<<<<<<< SEARCH
    def try_triple_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate three terms at once - more aggressive but still bounded."""
        if time_left() < 15:
            return U_list, V_list, W_list, rank
        
        max_triples = min(500, rank * (rank-1) * (rank-2) // 6)
        triples_checked = 0
=======
    def try_triple_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate three terms at once - more aggressive but still bounded."""
        if time_left() < 15:
            return U_list, V_list, W_list, rank
        
        max_triples = min(1000, rank * (rank-1) * (rank-2) // 6)
        triples_checked = 0
>>>>>>> REPLACE

