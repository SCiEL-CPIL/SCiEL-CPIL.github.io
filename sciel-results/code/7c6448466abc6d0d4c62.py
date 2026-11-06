<<<<<<< SEARCH
    def cut_size(labels):
        # Efficient calculation of cut size
        # Cut size = sum of edges where labels differ
        # We can compute this as: 0.5 * sum_{i,j} A[i,j] * (1 if labels[i] != labels[j] else 0)
        # Or simpler: for each i, sum A[i,j] where labels[j] != labels[i]
        # Then divide by 2 because each edge is counted twice.
        # Let's use matrix multiplication for speed if possible, or just loop for n=100.
        # n=100 is small, loop is fine.
        total = 0
        for i in range(n):
            if labels[i] == 0:
                total += A[i][labels == 1].sum()
            else:
                total += A[i][labels == 0].sum()
        return total / 2.0

    def local_search(labels):
        """Perform 1-flip local search until convergence."""
        improved = True
        # Use a random order to avoid bias
        indices = np.arange(n)
        
        while improved:
            improved = False
            np.random.shuffle(indices)
            for v in indices:
                # Count edges going to same side vs other side
                current_label = labels[v]
                other_label = 1 - current_label
                
                # Number of edges from v to other side (crossing)
                edges_crossing = A[v][labels == other_label].sum()
                # Number of edges from v to same side (non-crossing)
                edges_same = A[v][labels == current_label].sum()
                
                # If flipping v increases the cut (edges_same > edges_crossing), flip it
                # Gain = edges_same - edges_crossing
                if edges_same > edges_crossing:
                    labels[v] = other_label
                    improved = True
        
        return labels
=======
    def calculate_gains(labels):
        """Calculate the gain of flipping each vertex individually."""
        gains = np.zeros(n)
        for v in range(n):
            current_label = labels[v]
            # Gain = (edges to same side) - (edges to other side)
            # If we flip, edges to same become crossing, edges to crossing become same.
            # New cut contribution from v's edges = (old crossing)
            # Old cut contribution from v's edges = (old crossing)
            # Wait, gain in total cut size:
            # Delta = NewCut - OldCut
            # Edges (v, w):
            # If L_v == L_w, edge is internal (0). After flip, L_v != L_w, edge is crossing (1). Delta +1.
            # If L_v != L_w, edge is crossing (1). After flip, L_v == L_w, edge is internal (0). Delta -1.
            # So Delta = sum_{w in N(v)} (1 if L_v == L_w else -1)
            same = (labels == current_label)
            other = (labels == 1 - current_label)
            gains[v] = A[v][same].sum() - A[v][other].sum()
        return gains

    def local_search(labels):
        """Perform 1-flip and 2-flip local search until convergence."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        gains = calculate_gains(labels)
        
        improved = True
        while improved:
            improved = False
            
            # Try best 1-flip
            best_v = np.argmax(gains)
            if gains[best_v] > 0:
                labels[best_v] = 1 - labels[best_v]
                current_cut += gains[best_v]
                # Update gains for neighbors of best_v
                neighbors = np.where(A[best_v] > 0)[0]
                # Recalculate gains for neighbors and best_v itself
                # Actually, only neighbors' gains change, and best_v's gain changes.
                # Gain of u depends on labels of N(u).
                # If we flip v, gains of all u in N(v) change.
                # Also gain of v changes.
                for u in neighbors:
                    # Recalculate gain for u
                    current_label_u = labels[u]
                    same = (labels == current_label_u)
                    other = (labels == 1 - current_label_u)
                    gains[u] = A[u][same].sum() - A[u][other].sum()
                # Recalculate gain for v
                current_label_v = labels[best_v]
                same = (labels == current_label_v)
                other = (labels == 1 - current_label_v)
                gains[best_v] = A[best_v][same].sum() - A[best_v][other].sum()
                improved = True
                continue # Restart scan with updated gains

            # Try best 2-flip
            # Check all pairs (u, v)
            best_gain = 0
            best_u, best_v = -1, -1
            for u in range(n):
                if gains[u] <= 0:
                    continue # Optional optimization, but might miss combined gains? No, if both negative, sum likely negative. But interaction term matters.
                for v in range(u + 1, n):
                    # Calculate pair gain
                    # Delta_uv = g_u + g_v - 2 * A_uv * (1 if L_u==L_v else -1)
                    if labels[u] == labels[v]:
                        interaction = 1
                    else:
                        interaction = -1
                    pair_gain = gains[u] + gains[v] - 2 * A[u, v] * interaction
                    if pair_gain > best_gain:
                        best_gain = pair_gain
                        best_u, best_v = u, v
            
            if best_gain > 0:
                labels[best_u] = 1 - labels[best_u]
                labels[best_v] = 1 - labels[best_v]
                current_cut += best_gain
                # Update gains for neighbors of u and v, and u, v themselves
                neighbors = np.union1d(np.where(A[best_u] > 0)[0], np.where(A[best_v] > 0)[0])
                for u_idx in neighbors:
                    current_label_u = labels[u_idx]
                    same = (labels == current_label_u)
                    other = (labels == 1 - current_label_u)
                    gains[u_idx] = A[u_idx][same].sum() - A[u_idx][other].sum()
                # Ensure u and v are updated (they are in neighbors if self loops? No, but they are affected by their own flip? 
                # Actually, gain of u depends on labels of N(u). Flipping u changes L_u.
                # The formula for gain[u] uses labels of others.
                # Wait, my `calculate_gains` uses `labels` array.
                # If I flip u and v, the `labels` array is updated.
                # The gains for any node w in N(u) U N(v) might change because L_u or L_v changed.
                # The gains for u and v themselves might change because L_u and L_v changed?
                # Gain[u] = sum_{w in N(u)} (1 if L_u == L_w else -1).
                # If I flip u, L_u changes. So Gain[u] changes.
                # If I flip v, and v is in N(u), L_v changes. So Gain[u] changes.
                # So yes, update all neighbors and u, v.
                # My loop `for u_idx in neighbors` covers neighbors.
                # Are u and v in neighbors? Only if they have self-loops or are connected to each other?
                # A[u, v] > 0 implies v is in N(u). So v is in neighbors of u.
                # So v is updated. Is u updated? u is in neighbors of v. So u is updated.
                # So the loop covers u and v as well.
                
                improved = True
                continue # Restart scan

        return labels
>>>>>>> REPLACE

We also need to keep `cut_size` for the initial comparison and final return, but I removed it in the search block? No, I replaced `cut_size` and `local_search`. I need to make sure `cut_size` is still defined or I define it.
In the replacement, I used `current_cut` inside `local_search`. I need `cut_size` outside.
Let's look at the original code. `cut_size` is defined. My search block replaced `cut_size` and `local_search`.
I will keep `cut_size` as is, and only replace `local_search`.
Wait, the search block in my thought process included `cut_size`. I should just replace `local_search` and maybe add a helper `calculate_gains`.

Let's refine the edit.

1. Replace `local_search` with the new 1/2-flip version.
2. Add `calculate_gains` helper.
3. The `local_search` function now maintains `current_cut` internally but returns `labels`. The caller uses `cut_size` to evaluate. This is fine.

One detail: `np.where(A[best_v] > 0)[0]` creates a list of neighbors. This is efficient enough.

Let's double check the 2-flip gain formula.
$\Delta_{uv} = g_u + g_v - 2 A_{uv} \delta_{uv}$
where $\delta_{uv} = 1$ if $L_u = L_v$, $-1$ otherwise.
If $L_u = L_v$:
$g_u$ includes term for $v$: $+1$ (since $L_u=L_v$).
$g_v$ includes term for $u$: $+1$ (since $L_v=L_u$).
Sum $g_u+g_v$ includes $+2$ from edge $(u,v)$.
But in the new state, $L_u \neq L_v$ is false (they are both flipped, so still equal).
So edge $(u,v)$ remains internal.
The "gain" from edge $(u,v)$ should be 0.
But $g_u$ assumed $v$ stays put. If $v$ stays put and $u$ flips, $L_u$ becomes $1-L_u = 1-L_v \neq L_v$. So edge becomes crossing. Gain +1.
So $g_u$ has +1 from edge $(u,v)$.
$g_v$ has +1 from edge $(v,u)$.
So $g_u+g_v$ has +2 from this edge.
But actual change is 0.
So we must subtract 2.
Formula: $g_u + g_v - 2$.
My formula: $- 2 A_{uv} (1) = -2$. Correct.

If $L_u \neq L_v$:
$g_u$ includes term for $v$: $-1$ (since $L_u \neq L_v$).
$g_v$ includes term for $u$: $-1$ (since $L_v \neq L_u$).
Sum $g_u+g_v$ includes $-2$.
Actual change: $L_u$ flips, $L_v$ flips. $L_u \neq L_v \implies 1-L_u = L_v$ and $1-L_v = L_u$. Still different.
Edge remains crossing. Change 0.
So we must add 2.
Formula: $g_u + g_v - (-2) = g_u + g_v + 2$.
My formula: $- 2 A_{uv} (-1) = +2$. Correct.

The logic holds.

I will also increase `num_restarts` slightly or keep it at 20. Since local search is now more powerful (finds deeper optima), fewer restarts might be needed, but 20 is safe.

One optimization: In `local_search`, the 2-flip loop is $O(n^2)$. For $n=100$, 5000 iterations. Inside, simple arithmetic. This is very fast in Python?
5000 iterations * 20 restarts * (number of passes).
Number of passes depends on convergence. Usually < 10-20.
Total ops: $5000 \times 20 \times 20 = 2,000,000$ simple ops. This is well within 120s.

