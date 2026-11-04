"""QAP solver for tho150 using Linearized Assignment (LAP) Block Coordinate Descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Convert to float64 for fast arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    def compute_cost(p):
        """Compute exact cost using vectorized matrix multiplication."""
        return np.sum(a_f * b_f[np.ix_(p, p)])

    def lap_init(a_mat, b_mat):
        """
        Initialize using a greedy heuristic based on row sums, then refine with LAP.
        For a better initial guess, we can use the spectral method or just random.
        Let's try a simple heuristic: sort facilities by sum of flows, sort locations by sum of distances, and match.
        Actually, let's use the LAP on a heuristic cost matrix first to get a decent start.
        C_ij = sum_k A_ik * B_jk  (ignoring the permutation dependency for a moment, or using identity)
        This is equivalent to minimizing trace(A^T B P) roughly? No.
        Let's just use a random permutation as start, or a simple heuristic.
        Given time constraints, a random start is safe.
        """
        rng = np.random.RandomState(42)
        p = rng.permutation(n)
        return p

    def lap_bcd(p, current_cost, end_time):
        """
        Block Coordinate Descent on the Linearized Assignment Problem.
        
        At each step:
        1. Construct the interaction matrix W = A^T @ B.
           The QAP cost is sum_{i,j} A_{ij} B_{pi, pj} = sum_{i,j} (A^T B)_{ji} P_{pi, j} ... 
           Let's derive the linearization properly.
           
           Cost = sum_{i,j} A_{ij} B_{p_i, p_j}
           
           Let X be the assignment matrix (n x n), where X_{i, k} = 1 if facility i is at location k.
           Then B_{p_i, p_j} = sum_{k, l} B_{k, l} X_{i, k} X_{j, l}.
           Cost = sum_{i, j, k, l} A_{ij} B_{kl} X_{ik} X_{jl}.
           
           Linearization around current X (which is a permutation matrix P):
           Fix the "interaction" part.
           We want to solve for a new permutation Q that minimizes:
           sum_{i, j} A_{ij} B_{Q_i, Q_j}
           
           This is still quadratic.
           However, we can use the "Linearized Assignment Problem" heuristic where we approximate the quadratic term by a linear one based on the current solution.
           
           Consider the cost change if we move facility i to location k.
           The change depends on all other facilities j.
           Delta(i -> k) = sum_j A_{ij} (B_{k, p_j} - B_{p_i, p_j}) + sum_j A_{ji} (B_{p_j, k} - B_{p_j, p_i})
           
           This is O(n) per move.
           
           The LAP approach usually refers to solving:
           min sum_{i, k} C_{ik} X_{ik}
           where C_{ik} is a cost matrix derived from the current state.
           
           A common heuristic is:
           C_{ik} = sum_j A_{ij} B_{k, p_j} + sum_j A_{ji} B_{p_j, k}
           This represents the "interaction cost" of placing facility i at location k, given the current positions of all other facilities j.
           The term -A_{ii} B_{p_i, p_i} is constant for a given i? No, p_i changes.
           Actually, the standard linearization for QAP is:
           C_{ik} = sum_{j != i} A_{ij} B_{k, p_j} + sum_{j != i} A_{ji} B_{p_j, k}
           
           If we solve the assignment problem with this cost matrix, we get a new permutation.
           This is a "first-order" approximation or a greedy step.
           It is equivalent to one step of a specific local search where we re-optimize all facilities simultaneously based on the current neighbors.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        
        while time.time() < end_time:
            # Construct the linear cost matrix C
            # C[i, k] = sum_{j != i} A[i, j] B[k, p[j]] + sum_{j != i} A[j, i] B[p[j], k]
            
            # Term 1: sum_{j != i} A[i, j] B[k, p[j]]
            # Let M1 = A @ (B[:, p])^T ?
            # B[:, p] is the matrix B with columns permuted by p.
            # Let B_p = B[:, curr_p]  (n x n), where B_p[:, j] = B[:, curr_p[j]]
            # Then (B_p)[k, j] = B[k, curr_p[j]]
            # We want sum_j A[i, j] B_p[k, j] = (A @ B_p^T)[i, k]
            
            B_p = b_f[:, curr_p]  # n x n
            # B_p^T is n x n, (B_p^T)[j, k] = B_p[k, j] = B[k, curr_p[j]]
            # T1 = A @ B_p^T
            T1 = a_f @ B_p.T
            
            # Term 2: sum_{j != i} A[j, i] B[p[j], k]
            # We want sum_j A[j, i] B[curr_p[j], k]
            # Let B_p2 = B[curr_p, :]  (n x n), where B_p2[j, k] = B[curr_p[j], k]
            # Then sum_j A[j, i] B_p2[j, k] = (A.T @ B_p2)[i, k]
            
            B_p2 = b_f[curr_p, :]  # n x n
            T2 = a_f.T @ B_p2
            
            C = T1 + T2
            
            # Subtract the self-interaction terms which are not part of the "interaction with others"
            # The formula sum_{j != i} excludes j=i.
            # T1[i, k] includes j=i: A[i, i] B[k, curr_p[i]]
            # T2[i, k] includes j=i: A[i, i] B[curr_p[i], k]
            # So we subtract A[i, i] * (B[k, curr_p[i]] + B[curr_p[i], k]) from C[i, k]
            
            # Let's vectorize this subtraction
            # For each i, we subtract A[i, i] * (B[:, curr_p[i]] + B[curr_p[i], :]) from C[i, :]
            # This is tricky to vectorize directly as a matrix op without loops or advanced indexing.
            
            # Alternative: Keep the self terms?
            # If we keep them, C[i, k] includes A[i, i] B[k, curr_p[i]] + A[i, i] B[curr_p[i], k].
            # When we assign i to k, the actual cost contribution involving i is:
            # sum_j A[i, j] B[k, p_j] + sum_j A[j, i] B[p_j, k]
            # If we solve the assignment with C including self terms, it might bias the solution.
            # However, for a first pass, it's often fine. Let's try subtracting them properly.
            
            # Loop over i to subtract self terms (n=150, so 150 iterations is fast)
            for i in range(n):
                li = curr_p[i]
                # Subtract A[i, i] * B[k, li] from C[i, :]
                # And A[i, i] * B[li, k] from C[i, :]
                C[i, :] -= a_f[i, i] * (b_f[:, li] + b_f[li, :])
                
            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(C)
            new_p = np.empty(n, dtype=np.int64)
            new_p[row_ind] = col_ind
            
            # Calculate new cost
            new_cost = compute_cost(new_p)
            
            if new_cost < curr_cost:
                curr_p = new_p
                curr_cost = new_cost
            else:
                # If no improvement, we might need to break or try a perturbation.
                # Since this is a deterministic linearization, if it doesn't improve, we are likely at a local minimum of the linearized problem.
                # To escape, we can add a small random noise to C or break.
                # Given the time limit, breaking is safer unless we have a robust perturbation strategy.
                # Let's add a small random perturbation to encourage exploration.
                rng = np.random.RandomState(int(time.time() * 1000) % 100000)
                noise = rng.normal(0, 1, (n, n)) * 1e-3 * curr_cost
                C += noise
                row_ind, col_ind = linear_sum_assignment(C)
                new_p = np.empty(n, dtype=np.int64)
                new_p[row_ind] = col_ind
                new_cost = compute_cost(new_p)
                
                if new_cost < curr_cost:
                    curr_p = new_p
                    curr_cost = new_cost
                else:
                    # Stuck. Break out to save time for other methods or just return.
                    break
                    
        return curr_p, curr_cost

    # 1. Initialization
    p_init = lap_init(a_f, b_f)
    cost_init = compute_cost(p_init)
    
    best_p = p_init.copy()
    best_cost = cost_init
    
    # 2. LAP BCD Refinement
    end_time_lk = deadline - 5.0
    p_lk, cost_lk = lap_bcd(best_p, best_cost, end_time_lk)
    if cost_lk < best_cost:
        best_p = p_lk
        best_cost = cost_lk
        
    # 3. If time remains, try a second restart with a different random seed
    if time.time() < deadline - 2.0:
        rng_perturb = np.random.RandomState(123)
        p_pert = rng_perturb.permutation(n)
        cost_pert = compute_cost(p_pert)
        p_lk2, cost_lk2 = lap_bcd(p_pert, cost_pert, deadline - 0.5)
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2

    return best_p.tolist()
# EVOLVE-BLOCK-END
