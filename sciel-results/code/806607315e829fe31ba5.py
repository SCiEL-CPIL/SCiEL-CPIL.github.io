import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    n = 20
    # Precompute indices for pairs i < j
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pair_indices)
    
    # Map pair (i, j) to index in pair_indices for faster access? 
    # Or just use matrix comm[i, j]
    
    def calculate_penalty(comm):
        # comm is n x n matrix of common neighbors
        # Penalty is sum over i < j of max(0, comm[i, j] - 2)
        # We only care about upper triangle
        upper = np.triu(comm, k=1)
        excess = np.maximum(0, upper - 2)
        return np.sum(excess)

    def greedy_init():
        A = np.zeros((n, n), dtype=np.int8)
        comm = np.zeros((n, n), dtype=np.int32)
        
        # Random order of edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u, v) is valid
            # Common neighbors of u and v is comm[u, v]
            # If comm[u, v] >= 2, adding edge doesn't change comm[u, v] but changes others?
            # Wait, the constraint is on the FINAL graph.
            # If we add (u, v), we must ensure that NO pair has > 2 common neighbors.
            # The pairs that change are (u, w) and (v, w).
            # We need to check if any of these will exceed 2.
            
            # Potential new common neighbors for (u, w) is comm[u, w] + A[v, w]
            # Potential new common neighbors for (v, w) is comm[v, w] + A[u, w]
            
            # We can vectorize this check
            # new_comm_u = comm[u, :] + A[v, :]
            # new_comm_v = comm[v, :] + A[u, :]
            
            # But we only care about w != u, v for the constraint?
            # The constraint is "no two vertices have three common neighbours".
            # This applies to all pairs.
            
            # Let's just check the max value in the updated rows.
            # Note: comm[u, u] and comm[v, v] are 0.
            
            # Optimization: 
            # If we add (u, v):
            # For all w, comm[u, w] increases by A[v, w].
            # For all w, comm[v, w] increases by A[u, w].
            
            # We need max(comm[u, :] + A[v, :]) <= 2 ? No, we need max over w != u, v.
            # Actually, comm[u, v] is the number of common neighbors of u and v.
            # If comm[u, v] > 2, it's already invalid? No, comm[u, v] doesn't change when adding (u, v).
            # Wait, if comm[u, v] is currently 2, and we add (u, v), does it become 3?
            # No, common neighbors are vertices w connected to BOTH.
            # The edge (u, v) connects u and v. It doesn't create a new common neighbor w.
            # So comm[u, v] is invariant.
            
            # However, comm[u, w] for w in N(v) increases by 1.
            # So if comm[u, w] was 2, it becomes 3. Invalid.
            
            # Check:
            # max_val_u = np.max(comm[u, :] + A[v, :])
            # max_val_v = np.max(comm[v, :] + A[u, :])
            # But we must exclude the diagonal? 
            # comm[u, u] is 0. A[v, u] is 0 (before adding). So 0+0=0.
            # comm[u, v] is count. A[v, v] is 0. So comm[u, v] + 0 = comm[u, v].
            
            # So if np.max(comm[u, :] + A[v, :]) <= 2 and np.max(comm[v, :] + A[u, :]) <= 2, it's valid.
            
            check_u = comm[u, :] + A[v, :]
            check_v = comm[v, :] + A[u, :]
            
            if np.max(check_u) <= 2 and np.max(check_v) <= 2:
                A[u, v] = 1
                A[v, u] = 1
                comm[u, :] += A[v, :]
                comm[v, :] += A[u, :]
                
        return A, comm

    def polish(A, comm):
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        check_u = comm[u, :] + A[v, :]
                        check_v = comm[v, :] + A[u, :]
                        if np.max(check_u) <= 2 and np.max(check_v) <= 2:
                            A[u, v] = 1
                            A[v, u] = 1
                            comm[u, :] += A[v, :]
                            comm[v, :] += A[u, :]
                            improved = True
        return A, comm

    best_count = 0
    best_A = np.zeros((n, n), dtype=np.int8)

    def run_sa():
        nonlocal best_count, best_A
        
        A, comm = greedy_init()
        A, comm = polish(A, comm)
        
        cur_count = np.sum(A) // 2
        if cur_count > best_count:
            best_count = cur_count
            best_A = A.copy()
            
        T = 1.0
        alpha = 0.9999
        min_T = 0.01
        penalty_weight = 10.0 # Weight for violations in energy
        
        # Energy = -count + penalty_weight * sum(max(0, comm[i,j]-2))
        # We track delta energy.
        
        while time.time() < deadline - 2:
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            is_add = (A[u, v] == 0)
            
            # Calculate delta energy
            # Delta count = +1 if add, -1 if remove
            delta_count = 1 if is_add else -1
            
            # Delta penalty
            # If add:
            # new_comm_u = comm[u, :] + A[v, :]
            # new_comm_v = comm[v, :] + A[u, :]
            # If remove:
            # new_comm_u = comm[u, :] - A[v, :]  (Note: A[v, u] is 1, so we subtract 1 from comm[u, u]? No)
            # When removing (u, v):
            # A[u, v] becomes 0.
            # comm[u, w] decreases by A[v, w] (old value).
            # Since A[v, u] is 1, comm[u, u] decreases by 1? 
            # But comm[u, u] is always 0. 
            # Let's trace: comm[u, u] = sum_k A[u, k] A[u, k] = sum_k A[u, k] = degree(u).
            # Wait, my definition of comm was sum_k A[u, k] A[v, k].
            # So comm[u, u] = degree(u).
            # The constraint is on common neighbors of DISTINCT vertices.
            # So we should ignore diagonal in penalty.
            
            # Let's refine the penalty calculation to only consider i < j.
            # But for delta, we can just compute the change in the sum over i < j.
            
            if is_add:
                # Pairs affected: (u, w) for all w, (v, w) for all w.
                # We need to calculate the change in penalty for these pairs.
                # Penalty term for pair (a, b) is max(0, comm[a, b] - 2).
                
                # For pair (u, w):
                # old_val = comm[u, w]
                # new_val = comm[u, w] + A[v, w]
                # delta_p = max(0, new_val - 2) - max(0, old_val - 2)
                
                # Vectorized delta penalty for row u
                old_u = comm[u, :]
                new_u = old_u + A[v, :]
                delta_p_u = np.sum(np.maximum(0, new_u - 2) - np.maximum(0, old_u - 2))
                
                # Vectorized delta penalty for row v
                old_v = comm[v, :]
                new_v = old_v + A[u, :]
                delta_p_v = np.sum(np.maximum(0, new_v - 2) - np.maximum(0, old_v - 2))
                
                # We double counted pair (u, v)?
                # In delta_p_u, we included w=v. 
                # old_u[v] = comm[u, v]. new_u[v] = comm[u, v] + A[v, v] = comm[u, v].
                # So delta is 0 for w=v in row u.
                # In delta_p_v, we included w=u.
                # old_v[u] = comm[v, u]. new_v[u] = comm[v, u] + A[u, u] = comm[v, u].
                # So delta is 0 for w=u in row v.
                # So no double counting of the (u, v) pair itself in terms of change?
                # Wait. The penalty is sum over i < j.
                # My delta_p_u sums over ALL w.
                # This includes w < u and w > u.
                # This is fine, as long as we are consistent.
                # The total penalty is sum over i < j.
                # If we sum changes over all pairs (u, w) and (v, w), we cover all affected pairs.
                # Pair (u, w) is affected. Pair (v, w) is affected.
                # If w < u, pair is (w, u). It is included in delta_p_u (as index w).
                # So this is correct.
                
                total_delta_p = delta_p_u + delta_p_v
            else:
                # Remove
                # new_u = old_u - A[v, :]
                # Note: A[v, u] is 1. So new_u[u] = old_u[u] - 1.
                # comm[u, u] is degree(u). Removing edge (u, v) reduces degree(u) by 1.
                # So comm[u, u] decreases by 1.
                # But we don't care about diagonal in penalty.
                # However, np.max or sum will include it.
                # We should mask out the diagonal or ensure it doesn't contribute.
                # Since penalty is max(0, comm-2), and comm[u, u] >= 0,
                # if comm[u, u] > 2, it contributes.
                # But comm[u, u] is degree. Max degree is 19.
                # So diagonal DOES contribute to the penalty sum if we sum over all.
                # But the constraint is only for i != j.
                # So we must EXCLUDE diagonal from penalty calculation.
                
                # Let's create a mask for i < j or just ignore diagonal.
                # Easier: compute delta for all, then subtract diagonal contribution change.
                # Diagonal contribution for u: max(0, comm[u, u] - 2).
                # Change in comm[u, u] is -1 (since A[v, u]=1).
                # So delta_diag_u = max(0, new_u[u] - 2) - max(0, old_u[u] - 2).
                # We want to exclude this from the total penalty delta.
                
                old_u = comm[u, :]
                new_u = old_u - A[v, :]
                delta_p_u = np.sum(np.maximum(0, new_u - 2) - np.maximum(0, old_u - 2))
                
                old_v = comm[v, :]
                new_v = old_v - A[u, :]
                delta_p_v = np.sum(np.maximum(0, new_v - 2) - np.maximum(0, old_v - 2))
                
                # Subtract diagonal changes
                # For u:
                old_diag_u = old_u[u]
                new_diag_u = new_u[u]
                delta_diag_u = max(0, new_diag_u - 2) - max(0, old_diag_u - 2)
                
                # For v:
                old_diag_v = old_v[v]
                new_diag_v = new_v[v]
                delta_diag_v = max(0, new_diag_v - 2) - max(0, old_diag_v - 2)
                
                total_delta_p = (delta_p_u - delta_diag_u) + (delta_p_v - delta_diag_v)
            
            delta_E = -delta_count + penalty_weight * total_delta_p
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / max(T, min_T)):
                # Accept
                if is_add:
                    A[u, v] = 1
                    A[v, u] = 1
                    comm[u, :] += A[v, :]
                    comm[v, :] += A[u, :]
                else:
                    A[u, v] = 0
                    A[v, u] = 0
                    comm[u, :] -= A[v, :]
                    comm[v, :] -= A[u, :]
                
                cur_count += delta_count
                
                if cur_count > best_count:
                    best_count = cur_count
                    best_A = A.copy()
                    
            T = max(T * alpha, min_T)

    # Run multiple SA restarts
    for i in range(10):
        if time.time() > deadline - 5:
            break
        run_sa()
        
        # Final polish on best
        # We don't easily have the comm for best_A, so we reconstruct or just return best_A
        # Actually, let's polish the current best_A by reconstructing comm?
        # Or just run one more greedy on best_A?
        # Greedy on existing graph:
        # We can try to add edges to best_A.
        
        # Let's do a final polish on the best solution found so far.
        # Reconstruct comm for best_A
        best_comm = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                best_comm[i, j] = np.dot(best_A[i, :], best_A[j, :])
        
        # Polish best_A
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if best_A[u, v] == 0:
                        check_u = best_comm[u, :] + best_A[v, :]
                        check_v = best_comm[v, :] + best_A[u, :]
                        if np.max(check_u) <= 2 and np.max(check_v) <= 2:
                            best_A[u, v] = 1
                            best_A[v, u] = 1
                            best_comm[u, :] += best_A[v, :]
                            best_comm[v, :] += best_A[u, :]
                            improved = True
                            best_count += 1

    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j]:
                res.append([i, j])
    return res

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    print("Valid" if valid else "Invalid")
