    def get_delta_violations(i, j, c, is_add, adj):
        """
        Returns the change in violation count for color c
        when edge (i, j) is added (is_add=True) or removed (is_add=False).
        """
        dv = 0
        # Pairs (i, w)
        for w in range(n):
            if w == i or w == j: continue
            # Check if (w, j) is in color c
            if (adj[c][w] >> j) & 1:
                # j is a neighbor of w in c
                # For pair (i, w):
                # If is_add: N_c(i) gains j.
                #   If j in N_c(w), then j is a new common neighbor of i and w?
                #   Only if j was not already in N_c(i). It wasn't.
                #   So common count increases by 1.
                #   old_k = |N_c(i) \cap N_c(w)|
                #   new_k = old_k + 1
                #   delta_v = max(0, new_k-1) - max(0, old_k-1)
                #   If old_k >= 1: delta = 1.
                #   If old_k == 0: delta = 0.
                if is_add:
                    old_k = (adj[c][i] & adj[c][w]).bit_count()
                    if old_k >= 1:
                        dv += 1
                else:
                    # If is_remove: N_c(i) loses j.
                    #   If j in N_c(w), then j was a common neighbor.
                    #   So common count decreases by 1.
                    #   old_k = |N_c(i) \cap N_c(w)|
                    #   new_k = old_k - 1
                    #   delta_v = max(0, new_k-1) - max(0, old_k-1)
                    #   If old_k == 1: delta = 0 - 0 = 0.
                    #   If old_k == 2: delta = 0 - 1 = -1.
                    #   If old_k > 2: delta = (old_k-2) - (old_k-1) = -1.
                    old_k = (adj[c][i] & adj[c][w]).bit_count()
                    if old_k >= 2:
                        dv -= 1

        # Pairs (j, w)
        for w in range(n):
            if w == i or w == j: continue
            if (adj[c][w] >> i) & 1:
                if is_add:
                    old_k = (adj[c][j] & adj[c][w]).bit_count()
                    if old_k >= 1:
                        dv += 1
                else:
                    old_k = (adj[c][j] & adj[c][w]).bit_count()
                    if old_k >= 2:
                        dv -= 1
        return dv
