<<<<<<< SEARCH
    def find_max_biclique(remaining_mask, adj_mask, n):
        """Find a large biclique in the subgraph. remaining_mask is a set of edges represented as frozensets."""
        if not remaining_mask:
            return None

        # Build remaining adjacency
        rem_adj = [0] * n
        for (u, v) in remaining_mask:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)

        best_L = 0
        best_R = 0
        best_size = 0

        # Try each vertex as potential left singleton
        for u in range(n):
            # R candidates: neighbors of u in remaining
            R_cand = rem_adj[u]
            if R_cand == 0:
                continue

            # L candidates: vertices connected to all in R_cand
            # Start with u, then expand
            L = 1 << u
            R = R_cand

            # Expand to fixpoint
            changed = True
            while changed:
                changed = False
                # Find vertices connected to all in R
                common = all_mask
                for r in range(n):
                    if R & (1 << r):
                        common &= rem_adj[r]
                # Remove L and R from common
                new_L_cand = common & ~(L | R)
                if new_L_cand:
                    L |= new_L_cand
                    changed = True

                # Find vertices connected to all in L
                common = all_mask
                for l in range(n):
                    if L & (1 << l):
                        common &= rem_adj[l]
                new_R_cand = common & ~(L | R)
                if new_R_cand:
                    R |= new_R_cand
                    changed = True

            # Verify
            valid = True
            for l in range(n):
                if L & (1 << l):
                    for r in range(n):
                        if R & (1 << r):
                            if l == r:
                                continue
                            if not (rem_adj[l] & (1 << r)):
                                valid = False
                                break
                if not valid:
                    break

            if valid:
                size = popcount(L) * popcount(R)
                if size > best_size:
                    best_size = size
                    best_L = L
                    best_R = R

        if best_size == 0:
            return None
        return (bits_to_set(best_L), bits_to_set(best_R))
=======
    def find_max_biclique(remaining_mask, adj_mask, n):
        """Find a large biclique in the subgraph using edge seeds and expansion."""
        if not remaining_mask:
            return None

        # Build remaining adjacency
        rem_adj = [0] * n
        for (u, v) in remaining_mask:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)

        best_L = 0
        best_R = 0
        best_size = 0

        # Try each edge as a seed for biclique expansion
        for (u, v) in remaining_mask:
            # Start with L={u}, R={v}
            L = 1 << u
            R = 1 << v

            # Expand to fixpoint
            changed = True
            while changed:
                changed = False
                # Find vertices connected to all in R
                common = all_mask
                for r in range(n):
                    if R & (1 << r):
                        common &= rem_adj[r]
                new_L_cand = common & ~(L | R)
                if new_L_cand:
                    L |= new_L_cand
                    changed = True

                # Find vertices connected to all in L
                common = all_mask
                for l in range(n):
                    if L & (1 << l):
                        common &= rem_adj[l]
                new_R_cand = common & ~(L | R)
                if new_R_cand:
                    R |= new_R_cand
                    changed = True

            # Verify
            valid = True
            for l in range(n):
                if L & (1 << l):
                    for r in range(n):
                        if R & (1 << r):
                            if l == r:
                                continue
                            if not (rem_adj[l] & (1 << r)):
                                valid = False
                                break
                    if not valid:
                        break

            if valid:
                size = popcount(L) * popcount(R)
                if size > best_size:
                    best_size = size
                    best_L = L
                    best_R = R

        if best_size == 0:
            return None
        return (bits_to_set(best_L), bits_to_set(best_R))
>>>>>>> REPLACE

