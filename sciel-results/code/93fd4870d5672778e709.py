# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Strategy:
    1. Fix the first job as the one with the highest total processing time.
    2. Treat the remaining sequence as a TSP-like problem on a directed graph.
    3. Edge weights are approximated by the incremental makespan contribution.
    4. Use a greedy nearest-neighbor heuristic with 2-opt local search on the TSP tour.
    
    For larger instances, we use a fast greedy construction followed by limited local search
    to stay within time limits.
    """
    p = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = p.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Step 1: Fix the first job (highest total processing time)
    total_pt = p.sum(axis=1)
    first_job = int(np.argmax(total_pt))
    
    remaining_jobs = [j for j in range(n_jobs) if j != first_job]
    m = len(remaining_jobs)  # number of remaining jobs to sequence
    
    if m == 0:
        return [first_job]
    
    # Step 2: Build incremental cost matrix for inserting job j after job i
    # For the TSP approximation, we define edge cost c[i][j] as the estimated
    # incremental makespan when job j is placed after job i in the sequence.
    
    # Precompute pairwise completion time contributions
    # For a flow shop, the makespan depends on the cumulative load.
    # A good approximation for edge weight: 
    #   c[i][j] = max over machines k of (completion time contribution)
    # Simplified: use the sum of processing times weighted by machine position
    
    # Better approach: compute for each pair (i, j) the "cost" of placing j after i
    # This is approximated by: sum over machines of p[j][k] * (k+1) adjusted for i
    # Actually, let's use a simpler but effective heuristic:
    # c[i][j] = p[j].sum() + max_k(p[i][k])  -- rough but captures interaction
    
    # More principled: the incremental makespan when adding j after i in a sequence
    # depends on the current completion times. Since we're building a TSP graph,
    # we approximate edge weight as:
    #   w[i][j] = sum_k p[j][k] * (k + 1)  -- position-weighted load of j
    # plus a small interaction term with i
    
    # Let's use: w[i][j] = p[j].sum() + 0.5 * sum_k min(p[i][k], p[j][k])
    # This captures that similar jobs can be overlapped somewhat
    
    # Actually, for a cleaner TSP formulation, let's define:
    #   w[i][j] = sum_k p[j][k] * (k+1)   (position-weighted processing of j)
    # The idea: jobs with large processing on later machines should come earlier
    # to allow overlap with subsequent jobs on earlier machines.
    
    # Precompute position-weighted processing times for each job
    weights = np.arange(1, n_machines + 1, dtype=np.float64)
    pos_weighted_pt = p @ weights  # for each job, sum of p[j][k] * (k+1)
    
    # Edge cost matrix for the TSP on remaining jobs
    # w[i_idx][j_idx] = cost of going from job i to job j
    # We use: pos_weighted_pt[j] as the primary cost, plus a small interaction
    interaction = 0.01 * (p[remaining_jobs[:, None, :] * 0] if False else 0)  # placeholder
    
    # Simpler and more effective: use the NEH-style but in a TSP greedy framework
    # Let's build the TSP graph where edge weight from i to j is:
    #   w(i,j) = pos_weighted_pt[j] + 0.1 * p[i].sum() * p[j].sum() / (p.sum() + 1e-9)
    
    pt_sums = p.sum(axis=1)
    
    # Build cost matrix for remaining jobs
    rem_idx = np.array(remaining_jobs)
    n_rem = m
    
    # Cost matrix: cost of transitioning from job at index a to job at index b in rem_idx
    # w[a][b] = pos_weighted_pt[rem_idx[b]] + small_interaction[a][b]
    
    # For the TSP, we want to find a tour that minimizes total edge cost.
    # Use greedy nearest neighbor + 2-opt
    
    # Initialize: start from first_job, then build sequence greedily
    # Greedy: at each step, pick the unvisited job j that minimizes the "cost" of adding it
    
    # Define the incremental cost function for adding job j to the end of current sequence
    # For a flow shop, a good approximation of makespan when appending j to sequence S:
    #   makespan ≈ max over machines of (cumulative completion + p[j][k])
    
    # Let's use a proper greedy construction:
    # Start with first_job, then repeatedly add the job that minimizes the estimated makespan
    
    def estimate_makespan(sequence, p, n_machines):
        """Estimate makespan for a given job sequence."""
        if len(sequence) == 0:
            return 0.0
        # Compute completion times on each machine
        # C[k][j] = completion time of job j on machine k
        # C[0][j] = p[j][0]
        # C[k][j] = max(C[k-1][j], C[k][j-1]) + p[j][k]
        # We track only the last job's completion per machine
        last_comp = np.zeros(n_machines)
        for j in sequence:
            prev = 0.0
            for k in range(n_machines):
                prev = max(last_comp[k], prev) + p[j][k]
                last_comp[k] = prev
        return last_comp[-1]  # makespan is completion on last machine
    
    # Greedy construction
    current_seq = [first_job]
    unvisited = set(remaining_jobs)
    
    # Precompute for speed: for each unvisited job and current last job, estimate cost
    # Use a faster heuristic: sort by NEH key and use that as initial, then improve
    
    # Actually, let's use the TSP-inspired approach as specified:
    # Build a directed graph where edge weight from i to j approximates incremental makespan
    
    # For efficiency, precompute a cost matrix
    # cost[i][j] = estimated incremental makespan when j follows i
    # Approximation: max_k(p[i][k]) + p[j].sum() / 2  -- rough but fast
    
    # Better: use the actual makespan increment
    # When we have a partial sequence ending in i, and we add j, the makespan increases by
    # approximately: max_k(p[j][k] + max(0, C[k][i] - C[k][i-1])) where C is completion
    
    # For the TSP graph, let's use:
    #   w[i][j] = p[j].sum() + 0.5 * max_k(p[i][k])
    
    max_pt_per_job = p.max(axis=1)
    sum_pt_per_job = p.sum(axis=1)
    
    # Build NxN cost matrix for all jobs (including first_job)
    # w[i][j] = sum_pt_per_job[j] + 0.5 * max_pt_per_job[i]
    w = sum_pt_per_job[None, :] + 0.5 * max_pt_per_job[:, None]
    
    # Now solve TSP on remaining jobs, starting from first_job
    # Greedy nearest neighbor from first_job
    visited = {first_job}
    seq = [first_job]
    
    for _ in range(m):
        last = seq[-1]
        # Find unvisited job j that minimizes w[last][j]
        best_j = None
        best_cost = np.inf
        for j in unvisited:
            c = w[last][j]
            if c < best_cost:
                best_cost = c
                best_j = j
        seq.append(best_j)
        visited.add(best_j)
        unvisited.remove(best_j)
    
    # Now apply 2-opt local search on the TSP tour (excluding the fixed first job)
    # The full sequence is: first_job + [remaining in order]
    # We can swap adjacent pairs in the remaining part
    
    def compute_makespan_fast(seq, p, n_machines):
        """Compute makespan for a job sequence."""
        if not seq:
            return 0.0
        last_comp = np.zeros(n_machines)
        for j in seq:
            prev = 0.0
            for k in range(n_machines):
                prev = max(last_comp[k], prev) + p[j][k]
                last_comp[k] = prev
        return last_comp[-1]
    
    current_makespan = compute_makespan_fast(seq, p, n_machines)
    
    # 2-opt on the subsequence after first_job
    # Try reversing segments in the remaining part
    improved = True
    max_iters = 200
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        # Try all pairs of positions in the remaining part (indices 1 to n_jobs-1 in seq)
        for i in range(1, n_jobs - 1):
            for j in range(i + 1, n_jobs):
                # Reverse seq[i:j+1]
                new_seq = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
                new_makespan = compute_makespan_fast(new_seq, p, n_machines)
                if new_makespan < current_makespan - 1e-9:
                    seq = new_seq
                    current_makespan = new_makespan
                    improved = True
    
    return seq
# EVOLVE-BLOCK-END
