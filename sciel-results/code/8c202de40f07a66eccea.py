"""Rewritten program for Maximum Cut using a direct structural approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    """
    Assigns 0/1 labels to vertices to maximize the number of edges crossing the partition.
    Uses a combination of:
      1. Spectral initialization (Laplacian eigenvector)
      2. Fast local search (greedy hill-climbing) with vectorized gains
      3. Quantum-inspired annealing for escape from local optima
      4. Simulated annealing with vectorized gains
      5. Multi-start with random restarts
      6. Final high-precision polish pass
    Deadline guard ensures return well before 150s.
    """
    try:
        adj = np.asarray(adjacency, dtype=np.int32)
        n = adj.shape[0]
        if n == 0:
            return []

        # Ensure symmetric and no self loops for safety
        adj = (adj + adj.T) // 2
        np.fill_diagonal(adj, 0)

        start_time = time.time()
        DEADLINE = start_time + 140.0  # Safety margin before 150s kill

        def time_left():
            return DEADLINE - time.time()

        # Precompute
        degrees = adj.sum(axis=1)
        E = int(degrees.sum()) // 2
        adj_float = adj.astype(np.float64)
        degrees_float = degrees.astype(np.float64)

        # Cut size: number of edges with endpoints in different partitions
        def cut_size(labels):
            """Fast cut size: count edges crossing the cut."""
            diff = labels[:, None] != labels[None, :]
            return int(np.sum(adj * diff)) // 2

        def compute_gains(labels):
            """Vectorized computation of gain for flipping each vertex.
            gain[v] = 2 * (sum of adj[v,u] where labels[u] == labels[v]) - degrees[v]
            If gain > 0, flipping v increases the cut."""
            l0_mask = (labels == 0).astype(np.int8)
            l1_mask = (labels == 1).astype(np.int8)
            
            # For vertices with label 0: same_count = adj @ l0_mask
            # For vertices with label 1: same_count = adj @ l1_mask
            adj_l0 = adj_float @ l0_mask
            adj_l1 = adj_float @ l1_mask
            
            same_count = np.where(labels == 0, adj_l0, adj_l1)
            gains = 2.0 * same_count - degrees_float
            return gains

        def single_gain(labels, v):
            """Compute gain for flipping vertex v only. O(n) instead of O(n^2)."""
            # Count neighbors of v with same label as v
            same_label = labels[v]
            same_neighbors = int(np.sum(adj[v] * (labels == same_label)))
            gain = 2 * same_neighbors - int(degrees[v])
            return gain

        def local_search(labels, max_rounds=500, include_pairwise=False):
            """Greedy hill-climbing with vectorized gain computation via tracked opposite-side degrees.
            Optionally includes pairwise flip moves to escape local optima."""
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            
            # Track: opp_deg[v] = number of neighbors of v on the OPPOSITE side
            # gain[v] = 2*opp_deg[v] - deg[v] (flipping v changes same->opp and opp->same)
            # Actually: if v is on side 0, opp_deg[v] = adj[v] @ (labels==1)
            # After flip, new opp_deg[v] = deg[v] - opp_deg[v]
            # Gain of flip = (new_opp_deg[v] - opp_deg[v]) * 2 + sum over neighbors u of (change in opp_deg[u])
            # For neighbor u of v: if u is on same side as v, u loses one opp-neighbor (v was opp, now same)
            #   if u is on opposite side, u gains one opp-neighbor (v was same, now opp)
            # So: delta_opp_deg[u] = +1 if u opposite to v, -1 if u same as v
            # Total gain = 2*(deg[v] - 2*opp_deg[v]) + sum_u adj[v,u]*delta_opp_deg[u]
            #            = 2*deg[v] - 4*opp_deg[v] + (opp_deg[v] - (deg[v] - opp_deg[v]))
            #            = 2*deg[v] - 4*opp_deg[v] + 2*opp_deg[v] - deg[v]
            #            = deg[v] - 2*opp_deg[v]
            # Wait, let me re-derive: 
            # Current cut contribution involving v: opp_deg[v] edges cross
            # After flip: (deg[v] - opp_deg[v]) edges cross
            # But also neighbors' cut contributions change:
            #   For each neighbor u: if u was on same side as v, the edge (u,v) was NOT crossing, now IS crossing: +1
            #   If u was on opposite side, the edge (u,v) WAS crossing, now IS NOT: -1
            # Net change from neighbors' perspective: (same_side_neighbors) - (opp_side_neighbors) = (deg[v]-opp_deg[v]) - opp_deg[v] = deg[v] - 2*opp_deg[v]
            # Net change from v's perspective: (deg[v] - opp_deg[v]) - opp_deg[v] = deg[v] - 2*opp_deg[v]
            # Total gain = (deg[v] - 2*opp_deg[v]) + (deg[v] - 2*opp_deg[v]) = 2*(deg[v] - 2*opp_deg[v])
            # Hmm, that doesn't seem right either. Let me just use the direct formula.
            # 
            # Direct: gain = 2*(same_side_count) - deg[v] where same_side_count = deg[v] - opp_deg[v]
            #         = 2*deg[v] - 2*opp_deg[v] - deg[v] = deg[v] - 2*opp_deg[v]
            # 
            # So gain[v] = deg[v] - 2*opp_deg[v]
            
            # Compute initial opp_deg
            l0 = (best_labels == 0).astype(np.int32)
            l1 = (best_labels == 1).astype(np.int32)
            opp_deg = np.where(best_labels == 0, adj_float @ l1, adj_float @ l0)
            
            for rnd in range(max_rounds):
                if time_left() < 2.0:
                    break
                improved = False
                order = np.random.permutation(n)
                for v in order:
                    if time_left() < 1.0:
                        break
                    
                    # gain[v] = deg[v] - 2*opp_deg[v]
                    delta = int(degrees[v]) - 2 * int(opp_deg[v])
                    
                    if delta > 0:
                        old_label = best_labels[v]
                        new_label = 1 - old_label
                        best_labels[v] = new_label
                        
                        # Update opp_deg for v and its neighbors
                        # v's new opp_deg = deg[v] - old_opp_deg[v]
                        old_opp_v = int(opp_deg[v])
                        opp_deg[v] = degrees[v] - old_opp_v
                        
                        # For each neighbor u of v:
                        #   if u had same label as old v (i.e., u == old_label), then u loses one opp-neighbor
                        #   if u had opposite label to old v (i.e., u == new_label), then u gains one opp-neighbor
                        neighbors = np.nonzero(adj[v])[0]
                        for u in neighbors:
                            if best_labels[u] == new_label:
                                # u was on same side as v before flip (both old_label), now u is on opposite side from v
                                # Wait: v changed from old_label to new_label. u is on new_label side.
                                # Before: v was old_label, u was new_label -> opposite, so edge crossed
                                # After: v is new_label, u is new_label -> same, so edge doesn't cross
                                # u loses one opp-neighbor
                                opp_deg[u] -= 1
                            else:
                                # u was on old_label side (same as v before), now v is new_label (opposite to u)
                                # Before: same side, no crossing
                                # After: opposite side, crossing
                                # u gains one opp-neighbor
                                opp_deg[u] += 1
                        
                        best_cut += delta
                        improved = True
                
                # Pairwise flip moves to escape single-flip local optima
                if include_pairwise and not improved:
                    if time_left() < 2.0:
                        break
                    num_pair_tries = min(50, n * 2)
                    for _ in range(num_pair_tries):
                        if time_left() < 1.0:
                            break
                        v1 = np.random.randint(0, n)
                        v2 = np.random.randint(0, n)
                        if v1 == v2:
                            continue
                        
                        # Compute gain for flipping v1
                        g1 = int(degrees[v1]) - 2 * int(opp_deg[v1])
                        
                        # Simulate flip of v1 to get updated opp_deg
                        old_opp_v1 = int(opp_deg[v1])
                        old_label_v1 = best_labels[v1]
                        new_label_v1 = 1 - old_label_v1
                        opp_deg[v1] = degrees[v1] - old_opp_v1
                        neighbors_v1 = np.nonzero(adj[v1])[0]
                        for u in neighbors_v1:
                            if best_labels[u] == new_label_v1:
                                opp_deg[u] -= 1
                            else:
                                opp_deg[u] += 1
                        
                        # Compute gain for flipping v2 with updated opp_deg
                        g2 = int(degrees[v2]) - 2 * int(opp_deg[v2])
                        
                        # Revert flip of v1
                        opp_deg[v1] = old_opp_v1
                        for u in neighbors_v1:
                            if best_labels[u] == new_label_v1:
                                opp_deg[u] += 1
                            else:
                                opp_deg[u] -= 1
                        
                        delta_pair = g1 + g2
                        if delta_pair > 0:
                            # Apply flip of v1
                            best_labels[v1] = new_label_v1
                            opp_deg[v1] = degrees[v1] - old_opp_v1
                            for u in neighbors_v1:
                                if best_labels[u] == new_label_v1:
                                    opp_deg[u] -= 1
                                else:
                                    opp_deg[u] += 1
                            
                            # Apply flip of v2
                            old_opp_v2 = int(opp_deg[v2])
                            old_label_v2 = best_labels[v2]
                            new_label_v2 = 1 - old_label_v2
                            best_labels[v2] = new_label_v2
                            opp_deg[v2] = degrees[v2] - old_opp_v2
                            neighbors_v2 = np.nonzero(adj[v2])[0]
                            for u in neighbors_v2:
                                if best_labels[u] == new_label_v2:
                                    opp_deg[u] -= 1
                                else:
                                    opp_deg[u] += 1
                            
                            best_cut += delta_pair
                            improved = True
                
                if not improved:
                    break
            return best_labels, best_cut

        def simulated_annealing(labels, max_iters=50000, T_start=4.0, T_end=0.05, cooling=0.99997):
            """Simulated annealing to escape local optima with tracked opp_deg for fast gains."""
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            current_labels = labels.copy().astype(np.int8)
            current_cut = best_cut
            T = T_start
            
            # Track opp_deg for fast gain computation
            l0 = (current_labels == 0).astype(np.int32)
            l1 = (current_labels == 1).astype(np.int32)
            opp_deg = np.where(current_labels == 0, adj_float @ l1, adj_float @ l0)
            
            def flip_vertex(v):
                """Flip vertex v and update opp_deg. Returns the gain."""
                nonlocal current_cut
                old_opp_v = int(opp_deg[v])
                gain = int(degrees[v]) - 2 * old_opp_v
                old_label = current_labels[v]
                new_label = 1 - old_label
                current_labels[v] = new_label
                opp_deg[v] = degrees[v] - old_opp_v
                neighbors = np.nonzero(adj[v])[0]
                for u in neighbors:
                    if current_labels[u] == new_label:
                        opp_deg[u] -= 1
                    else:
                        opp_deg[u] += 1
                current_cut += gain
                return gain
            
            for it in range(max_iters):
                if time_left() < 3.0:
                    break
                
                # Occasionally try a 2-flip move to escape local optima
                if np.random.random() < 0.15:
                    v1 = np.random.randint(0, n)
                    v2 = np.random.randint(0, n)
                    if v1 == v2:
                        v2 = (v1 + 1) % n
                    
                    # Compute gain for flipping v1
                    g1 = int(degrees[v1]) - 2 * int(opp_deg[v1])
                    
                    # Simulate flip of v1
                    old_opp_v1 = int(opp_deg[v1])
                    old_label_v1 = current_labels[v1]
                    new_label_v1 = 1 - old_label_v1
                    opp_deg[v1] = degrees[v1] - old_opp_v1
                    neighbors_v1 = np.nonzero(adj[v1])[0]
                    for u in neighbors_v1:
                        if current_labels[u] == new_label_v1:
                            opp_deg[u] -= 1
                        else:
                            opp_deg[u] += 1
                    
                    # Compute gain for flipping v2 with updated state
                    g2 = int(degrees[v2]) - 2 * int(opp_deg[v2])
                    
                    # Revert flip of v1
                    opp_deg[v1] = old_opp_v1
                    for u in neighbors_v1:
                        if current_labels[u] == new_label_v1:
                            opp_deg[u] += 1
                        else:
                            opp_deg[u] -= 1
                    
                    delta = g1 + g2
                    if delta >= 0:
                        # Apply both flips
                        flip_vertex(v1)
                        flip_vertex(v2)
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_labels = current_labels.copy()
                    else:
                        if T > 0.001:
                            prob = np.exp(delta / T)
                            if np.random.random() < prob:
                                flip_vertex(v1)
                                flip_vertex(v2)
                else:
                    # Use vectorized gain to find candidate vertices
                    gains = degrees_float - 2.0 * opp_deg
                    # Pick a vertex: prefer high-gain, but allow random exploration
                    if T > 0.5:
                        # At high temperature, pick more randomly
                        v = np.random.randint(0, n)
                    else:
                        # At low temperature, bias toward high-gain vertices
                        # Use softmax-like selection
                        v = np.random.choice(n, p=np.exp(gains / max(T, 0.1)) / np.sum(np.exp(gains / max(T, 0.1))))
                    
                    delta = int(degrees[v]) - 2 * int(opp_deg[v])
                    
                    if delta >= 0:
                        flip_vertex(v)
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_labels = current_labels.copy()
                    else:
                        if T > 0.001:
                            prob = np.exp(delta / T)
                            if np.random.random() < prob:
                                flip_vertex(v)

                T = max(T * cooling, T_end)

            return best_labels, best_cut

        def quantum_annealing(labels, max_iters=80000, s_start=1.0, s_end=0.0, cooling=0.99998):
            """Quantum-inspired annealing: interpolate between transverse field and problem Hamiltonian.
            The transverse field strength s controls the effective temperature for accepting worse moves."""
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            current_labels = labels.copy().astype(np.int8)
            current_cut = best_cut
            s = s_start  # transverse field strength (1.0 = pure quantum, 0.0 = pure classical)
            
            for it in range(max_iters):
                if time_left() < 3.0:
                    break
                
                # Compute gains for the selected vertex
                v = np.random.randint(0, n)
                gains = compute_gains(current_labels)
                delta = int(gains[v])

                # Quantum tunneling: effective temperature scales with transverse field
                T_eff = s * 3.0 + 0.05  # minimum temperature to allow some exploration
                
                if delta >= 0:
                    # Accept
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += delta
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = current_labels.copy()
                else:
                    # Accept with probability exp(delta / T_eff)
                    if T_eff > 0.01:
                        prob = np.exp(delta / T_eff)
                        if np.random.random() < prob:
                            current_labels[v] = 1 - current_labels[v]
                            current_cut += delta

                # Update transverse field
                s = max(s * cooling, s_end)

            return best_labels, best_cut

        def spectral_init():
            """Use the Laplacian eigenvector to initialize the partition."""
            try:
                D = np.diag(degrees)
                L = D - adj_float
                from scipy.linalg import eigh
                evals, evecs = eigh(L, subset_by_index=[0, 1])
                v = evecs[:, 1]
                labels = (v > 0).astype(int)
                if labels.sum() == 0 or labels.sum() == n:
                    labels = np.random.randint(0, 2, size=n)
                return labels
            except Exception:
                return np.random.randint(0, 2, size=n)

        def greedy_sequential_init():
            """Greedy sequential initialization: assign vertices in order of degree,
            placing each vertex on the side that maximizes immediate cut contribution."""
            labels = np.full(n, -1, dtype=np.int8)  # -1 means unplaced
            order = np.argsort(-degrees)
            
            for v in order:
                if time_left() < 1.0:
                    break
                # Count placed neighbors on each side
                placed = labels >= 0
                adj_v = adj[v]
                
                # Neighbors of v that are already placed on side 0
                placed0 = placed & (labels == 0)
                placed1 = placed & (labels == 1)
                
                count0 = int(np.sum(adj_v * placed0))
                count1 = int(np.sum(adj_v * placed1))
                
                # Put v on the side with fewer placed neighbors (maximizes cut)
                if count0 < count1:
                    labels[v] = 0
                elif count1 < count0:
                    labels[v] = 1
                else:
                    labels[v] = np.random.randint(0, 2)
            
            # Any unplaced vertices get random labels
            unplaced = labels < 0
            if np.any(unplaced):
                labels[unplaced] = np.random.randint(0, 2, size=int(np.sum(unplaced)))
            
            return labels.astype(np.int8)

        # Strategy:
        # 1. Spectral initialization + local search
        # 2. Multiple random restarts with local search
        # 3. Quantum-inspired annealing from best solutions
        # 4. Simulated annealing from best solutions
        # 5. Final high-precision polish

        best_labels = None
        best_cut = -1

        # Phase 1: Spectral init + local search
        labels_spec = spectral_init()
        labels_ls, cut_ls = local_search(labels_spec, max_rounds=200)
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls.copy()

        # Phase 1b: Greedy init + local search
        labels_greedy = greedy_sequential_init()
        labels_gls, cut_gls = local_search(labels_greedy, max_rounds=200)
        if cut_gls > best_cut:
            best_cut = cut_gls
            best_labels = labels_gls.copy()

        # Phase 2: Random restarts with local search
        num_restarts = 40
        for r in range(num_restarts):
            if time_left() < 5.0:
                break
            labels_rand = np.random.randint(0, 2, size=n)
            labels_rs, cut_rs = local_search(labels_rand, max_rounds=100)
            if cut_rs > best_cut:
                best_cut = cut_rs
                best_labels = labels_rs.copy()

        # Phase 3: Simulated annealing from best (replacing slower quantum annealing)
        if time_left() > 15.0 and best_labels is not None:
            labels_sa1, cut_sa1 = simulated_annealing(best_labels, max_iters=150000, T_start=5.0, T_end=0.01, cooling=0.99998)
            if cut_sa1 > best_cut:
                best_cut = cut_sa1
                best_labels = labels_sa1.copy()
            # Polish after simulated annealing
            if time_left() > 5.0:
                labels_polish, cut_polish = local_search(best_labels, max_rounds=200)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()

        # Phase 4: Second simulated annealing from best with different parameters
        if time_left() > 15.0 and best_labels is not None:
            labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=100000, T_start=3.0, T_end=0.02, cooling=0.99999)
            if cut_sa > best_cut:
                best_cut = cut_sa
                best_labels = labels_sa.copy()
            # Polish after simulated annealing
            if time_left() > 5.0:
                labels_polish, cut_polish = local_search(best_labels, max_rounds=200)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()

        # Phase 4b: Additional SA rounds from current best if time permits
        if time_left() > 20.0 and best_labels is not None:
            labels_sa2, cut_sa2 = simulated_annealing(best_labels, max_iters=80000, T_start=2.0, T_end=0.01, cooling=0.99999)
            if cut_sa2 > best_cut:
                best_cut = cut_sa2
                best_labels = labels_sa2.copy()
            if time_left() > 3.0:
                labels_polish, cut_polish = local_search(best_labels, max_rounds=200)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()

        # Phase 5: More random restarts with SA if time permits
        if time_left() > 25.0:
            for r in range(10):
                if time_left() < 5.0:
                    break
                labels_rand = np.random.randint(0, 2, size=n)
                labels_sa3, cut_sa3 = simulated_annealing(labels_rand, max_iters=50000, T_start=4.0, T_end=0.02, cooling=0.99999)
                if cut_sa3 > best_cut:
                    best_cut = cut_sa3
                    best_labels = labels_sa3.copy()
                # Polish
                if time_left() > 3.0:
                    labels_polish, cut_polish = local_search(best_labels, max_rounds=100)
                    if cut_polish > best_cut:
                        best_cut = cut_polish
                        best_labels = labels_polish.copy()

        # Phase 6: Final high-precision polish with more iterations
        if time_left() > 3.0 and best_labels is not None:
            for _ in range(50):
                if time_left() < 1.5:
                    break
                labels_polish, cut_polish = local_search(best_labels, max_rounds=500)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()
                else:
                    break

        # Phase 7: Final pairwise polish to escape single-flip local optima
        if time_left() > 5.0 and best_labels is not None:
            labels_pw, cut_pw = local_search(best_labels, max_rounds=300, include_pairwise=True)
            if cut_pw > best_cut:
                best_cut = cut_pw
                best_labels = labels_pw.copy()
            # One more single-flip polish after pairwise
            if time_left() > 2.0:
                labels_final, cut_final = local_search(best_labels, max_rounds=300)
                if cut_final > best_cut:
                    best_cut = cut_final
                    best_labels = labels_final.copy()
            # Final SA burst if time remains
            if time_left() > 10.0:
                labels_sa_final, cut_sa_final = simulated_annealing(best_labels, max_iters=50000, T_start=1.5, T_end=0.01, cooling=0.99999)
                if cut_sa_final > best_cut:
                    best_cut = cut_sa_final
                    best_labels = labels_sa_final.copy()
                if time_left() > 2.0:
                    labels_final2, cut_final2 = local_search(best_labels, max_rounds=200)
                    if cut_final2 > best_cut:
                        best_cut = cut_final2
                        best_labels = labels_final2.copy()

        if best_labels is None:
            best_labels = np.random.randint(0, 2, size=n)

        return best_labels.tolist()

    except Exception as e:
        # Fallback to random partition if anything fails
        try:
            n = np.asarray(adjacency).shape[0]
            return np.random.randint(0, 2, size=n).tolist()
        except:
            return []
# EVOLVE-BLOCK-END
