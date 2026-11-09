"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    def compute_cut(labels):
        # labels: 0/1 int array
        # cut = 0.5 * sum_{i,j} adj[i,j] * (labels[i] != labels[j])
        # Equivalent: sum over i<j
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def greedy_1flip(labels, max_passes=10):
        """Greedy 1-flip local search. Returns improved labels."""
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            for v in range(n):
                if time.time() > deadline:
                    break
                same = adj[v][labels == labels[v]].sum()
                other = adj[v][labels != labels[v]].sum()
                if same > other:
                    labels[v] = 1 - labels[v]
                    changed = True
            if not changed:
                break
        return labels

    def fast_2swap(labels, max_passes=100):
        """Fast 2-swap local search using O(1) gain formula."""
        # Precompute degree to each side for each node
        # For node v with label l:
        #   same[v] = sum of adj[v, u] where labels[u] == l
        #   other[v] = sum of adj[v, u] where labels[u] != l
        # When we swap labels of i and j (where labels[i] != labels[j]):
        #   Let li = labels[i], lj = labels[j]
        #   For node i:
        #     Before: same[i] = sum adj[i,k] where labels[k]==li, other[i] = sum adj[i,k] where labels[k]==lj
        #     After swap, i has label lj:
        #       new_same[i] = sum adj[i,k] where labels[k]==lj (excluding j) + adj[i,j] if adj[i,j]==1
        #       Actually, let's use the gain formula directly.
        #     
        #   Gain of swapping i and j (labels[i] != labels[j]):
        #   Let a = adj[i,j]
        #   Let si = sum of adj[i,k] where labels[k] == labels[i]  (same-side neighbors of i)
        #   Let oi = sum of adj[i,k] where labels[k] == labels[j]  (other-side neighbors of i)
        #   Let sj = sum of adj[j,k] where labels[k] == labels[j]  (same-side neighbors of j)
        #   Let oj = sum of adj[j,k] where labels[k] == labels[i]  (other-side neighbors of j)
        #   
        #   Note: si includes a if a==1 (since labels[j]!=labels[i], j is on other side... wait)
        #   labels[i] = li, labels[j] = lj, li != lj
        #   si = sum_{k: labels[k]==li} adj[i,k]. j has label lj != li, so j is NOT in si.
        #   oi = sum_{k: labels[k]==lj} adj[i,k]. j has label lj, so j IS in oi if a==1.
        #   So oi includes a.
        #   Similarly, sj = sum_{k: labels[k]==lj} adj[j,k]. i has label li != lj, so i NOT in sj.
        #   oj = sum_{k: labels[k]==li} adj[j,k]. i has label li, so i IS in oj if a==1.
        #   So oj includes a.
        #
        #   After swap: i has label lj, j has label li.
        #   Change in cut for edges incident to i (excluding i-j):
        #     Neighbors with label li: were same (not cut), now other (cut). Gain: +si
        #     Neighbors with label lj (excluding j): were other (cut), now same (not cut). Gain: -(oi - a)
        #   Change for edges incident to j (excluding i-j):
        #     Neighbors with label lj: were same (not cut), now other (cut). Gain: +sj
        #     Neighbors with label li (excluding i): were other (cut), now same (not cut). Gain: -(oj - a)
        #   Edge (i,j): was cut (li!=lj), still cut (lj!=li). No change.
        #
        #   Total gain = si + sj - (oi - a) - (oj - a) = si + sj - oi - oj + 2*a
        #
        #   We can compute si, oi, sj, oj quickly using precomputed row sums per label.
        
        # Precompute for each node v:
        #   sum_same[v] = sum of adj[v, u] where labels[u] == labels[v]
        #   sum_other[v] = sum of adj[v, u] where labels[u] != labels[v]
        # Update these incrementally when we accept a swap.
        
        labels = labels.copy()
        # Initialize sum_same and sum_other
        sum_same = np.zeros(n, dtype=np.int32)
        sum_other = np.zeros(n, dtype=np.int32)
        for v in range(n):
            sum_same[v] = adj[v][labels == labels[v]].sum()
            sum_other[v] = adj[v][labels != labels[v]].sum()
        
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            # Shuffle order to avoid bias
            order = list(range(n))
            np.random.shuffle(order)
            for i_idx in range(n):
                if time.time() > deadline:
                    break
                i = order[i_idx]
                for j_idx in range(i_idx + 1, n):
                    if time.time() > deadline:
                        break
                    j = order[j_idx]
                    if labels[i] == labels[j]:
                        continue
                    
                    # Compute gain
                    a = int(adj[i, j])
                    si = int(sum_same[i])   # neighbors of i with same label as i
                    oi = int(sum_other[i])   # neighbors of i with other label (which is labels[j])
                    sj = int(sum_same[j])   # neighbors of j with same label as j
                    oj = int(sum_other[j])   # neighbors of j with other label (which is labels[i])
                    
                    gain = si + sj - oi - oj + 2 * a
                    
                    if gain > 0:
                        # Accept swap
                        labels[i], labels[j] = labels[j], labels[i]
                        # Update sum_same and sum_other
                        # For node i:
                        #   Before: label li, sum_same=si, sum_other=oi
                        #   After: label lj, 
                        #     new sum_same[i] = sum of adj[i,k] where labels[k]==lj
                        #       = oi - a  (neighbors with label lj, excluding j) + a (j itself, if a==1)
                        #       Wait: oi = sum of adj[i,k] where labels[k]==lj. This includes j if a==1.
                        #       So new sum_same[i] = oi
                        #     new sum_other[i] = sum of adj[i,k] where labels[k]!=lj = sum of adj[i,k] where labels[k]==li
                        #       = si
                        # For node j:
                        #   Before: label lj, sum_same=sj, sum_other=oj
                        #   After: label li,
                        #     new sum_same[j] = oj
                        #     new sum_other[j] = sj
                        # For other nodes k:
                        #   If labels[k] == li:
                        #     For node k, i was in "other" (since labels[i]==li==labels[k]... wait no)
                        #     Let me re-think.
                        #     Node k has label lk.
                        #     Before swap:
                        #       i has label li, j has label lj.
                        #       If lk == li: i is same-side for k, j is other-side for k.
                        #       If lk == lj: i is other-side for k, j is same-side for k.
                        #     After swap:
                        #       i has label lj, j has label li.
                        #       If lk == li: i is now other-side, j is now same-side.
                        #       If lk == lj: i is now same-side, j is now other-side.
                        #     So for each k:
                        #       If lk == li: sum_same[k] changes by -adj[k,i] + adj[k,j]
                        #       If lk == lj: sum_same[k] changes by +adj[k,i] - adj[k,j]
                        #       sum_other[k] = degree[k] - sum_same[k], so it updates accordingly.
                        #     
                        #     This is O(n) per swap, which is expensive. Let's just recompute for all nodes?
                        #     No, that's O(n^2). 
                        #     
                        #     Alternative: just recompute sum_same and sum_other for all nodes after each pass.
                        #     That's O(n^2) per pass, which is 10k ops. With 100 passes, that's 1M ops. Fine.
                        #     But we need them for the gain calculation within the pass.
                        #     
                        #     Let's just recompute at the start of each pass.
                        #     Actually, let's just not maintain them incrementally. Recompute per pass.
                        #     For now, let's accept the swap and mark that we need to recompute.
                        #     Simpler: recompute sum_same/sum_other at the start of each while iteration.
                        #     But we're inside the inner loop. Let's just recompute after the full pass.
                        #     For the gain calculation within the pass, we use the values from the start of the pass.
                        #     This is a slight inaccuracy but should still work.
                        
                        # Actually, let's just recompute sum_same and sum_other after each accepted swap for correctness.
                        # O(n^2) per swap is too slow if many swaps are accepted.
                        # 
                        # Let's use a different approach: don't precompute, just compute si, oi, sj, oj on the fly.
                        # si = adj[i][labels == labels[i]].sum() -- this is O(n)
                        # So gain computation is O(n) per pair. 5000 pairs * O(n) = 5000*100 = 500k ops per pass.
                        # That's fine.
                        
                        improved = True
                        # Recompute sum_same and sum_other for correctness in subsequent iterations
                        # This is O(n^2) but only done when a swap is accepted
                        for v in range(n):
                            sum_same[v] = adj[v][labels == labels[v]].sum()
                            sum_other[v] = adj[v][labels != labels[v]].sum()
        
        return labels

    # Strategy 1: Go-with-the-flow + 1-flip + 2-swap
    if time.time() < deadline:
        labels = np.arange(n) % 2
        labels = greedy_1flip(labels, max_passes=10)
        labels = fast_2swap(labels, max_passes=50)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Strategy 2: Random restarts with 1-flip + 2-swap
    for restart in range(1, 21):
        if time.time() > deadline:
            break
        labels = np.random.randint(0, 2, size=n)
        labels = greedy_1flip(labels, max_passes=5)
        labels = fast_2swap(labels, max_passes=30)
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final polish
    if best_labels is not None and time.time() < deadline:
        final_labels = best_labels.copy()
        final_labels = fast_2swap(final_labels, max_passes=50)
        final_cut = compute_cut(final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
