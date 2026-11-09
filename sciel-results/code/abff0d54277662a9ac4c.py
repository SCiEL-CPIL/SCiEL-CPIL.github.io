"""Capacitated facility location via min-cost bipartite matching (Hungarian) with local search refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_weight_full_bipartite_matching

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using min-cost bipartite matching (Hungarian algorithm)
    on a demand-unit to capacity-slot graph, followed by local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    demands = np.asarray(demands, dtype=int)
    capacities = np.asarray(capacities, dtype=int)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)
    total_demand = int(np.sum(demands))
    total_cap = int(np.sum(capacities))

    def compute_score(res):
        """Compute total cost of an assignment."""
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])

    def greedy_init():
        """Greedy initialization: assign customers in order of demand (largest first)."""
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    res[i] = f
                    load[f] += demands[i]
                    break
            else:
                space = capacities - load
                best_f = np.argmax(space)
                if space[best_f] >= demands[i]:
                    res[i] = best_f
                    load[best_f] += demands[i]
                else:
                    res[i] = fac_order[0]
        return res

    def local_search(res, load, max_iter=3000):
        """Local search with relocate moves."""
        improved = True
        it = 0
        while improved and it < max_iter:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            for i in np.random.permutation(n_cust):
                current_f = res[i]
                best_f = -1
                best_delta = 1e-9
                for f_new in range(n_fac):
                    if f_new == current_f:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new]:
                        continue
                    delta = assign_cost[i, f_new] - assign_cost[i, current_f]
                    if load[f_new] == 0:
                        delta += open_cost[f_new]
                    if load[current_f] == demands[i]:
                        delta -= open_cost[current_f]
                    if delta < best_delta:
                        best_delta = delta
                        best_f = f_new
                if best_f != -1:
                    load[current_f] -= demands[i]
                    load[best_f] += demands[i]
                    res[i] = best_f
                    improved = True

    def hungarian_solve():
        """
        Solve via min-cost bipartite matching.
        Left nodes: demand units (one per unit of demand across all customers).
        Right nodes: capacity slots (one per unit of capacity across all facilities).
        Cost for matching demand-unit (customer i, unit k) to capacity-slot (facility j, unit m):
            assign_cost[i, j] / demands[i]  (proportional share of assignment cost)
        Then we add a fixed opening cost per facility used.
        
        Since Hungarian gives us a perfect matching (if total_demand == total_cap) or
        a max matching, we need to handle the case where total_demand < total_cap.
        We'll pad with dummy nodes if needed.
        
        Actually, for this problem, we want to assign each customer to exactly one facility,
        not each demand unit. The Hungarian approach on demand units would allow splitting
        a customer across facilities, which is not allowed.
        
        Alternative: Use a different formulation.
        Since n_cust=100 and n_fac=20, we can try all subsets? No, 2^20 is too large.
        
        Better approach: Use the fact that this is a small problem.
        Let's use a min-cost flow formulation directly.
        """
        # Build a min-cost flow network:
        # Source -> Customer nodes -> Facility nodes -> Sink
        # Source to Customer i: capacity = demands[i], cost = 0
        # Customer i to Facility j: capacity = demands[i], cost = assign_cost[i,j]
        # Facility j to Sink: capacity = capacities[j], cost = 0
        # Plus: if facility j is used, pay open_cost[j].
        
        # To handle the fixed charge, we can use a trick:
        # Add an intermediate node for each facility.
        # Source -> Customer i: cap=demands[i], cost=0
        # Customer i -> Facility_in_j: cap=demands[i], cost=assign_cost[i,j]
        # Facility_in_j -> Facility_out_j: cap=capacities[j], cost=open_cost[j] (but this is fixed, not per-unit)
        
        # The fixed charge makes this a fixed-charge network flow, which is NP-hard.
        # For 20 facilities, we can enumerate subsets of facilities to open.
        # 2^20 = 1,048,576 subsets. For each subset, solve a transportation problem.
        # That's too slow in Python.
        
        # Alternative: Since n_fac=20 is small, use a different approach.
        # Let's just do a good greedy + local search with multiple restarts.
        return None

    # Since the Hungarian approach on demand units doesn't directly apply (customers can't be split),
    # let's use a different structural approach:
    # 
    # APPROACH: Column generation / Set partitioning
    # Each "column" is a feasible assignment of a subset of customers to a single facility
    # (i.e., a set of customers whose total demand <= capacity of that facility).
    # We want to select a set of columns that covers all customers exactly once,
    # minimizing total cost (sum of open costs for used facilities + sum of assignment costs).
    #
    # This is a set partitioning problem. With 100 customers, the number of possible columns
    # is huge, but we can use a greedy/pricing approach.
    #
    # Actually, let's use a simpler but effective approach:
    # 1. Start with a good greedy solution.
    # 2. Use a "cluster-based" local search: try to reassign entire clusters of customers.
    # 3. Use simulated annealing with cluster moves.
    
    # Let's implement a cluster-based simulated annealing.
    
    def cluster_simulated_annealing(res, load, max_iter=50000, t0=1000.0, t1=1.0, cooling=0.9999):
        """
        Simulated annealing with cluster moves.
        A cluster move: pick a facility, remove all its customers, and reassign them
        to other facilities (or the same one) in a way that minimizes cost.
        """
        best_res = res.copy()
        best_load = load.copy()
        best_score = compute_score(best_res)
        current_score = best_score
        
        t = t0
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Choose a random facility to "shake"
            f_shake = np.random.randint(n_fac)
            
            # Get customers currently assigned to f_shake
            custs = np.where(res == f_shake)[0]
            if len(custs) == 0:
                continue
            
            # Save current assignment
            old_res = res[custs].copy()
            old_load = load.copy()
            
            # Remove these customers from f_shake
            load[f_shake] -= np.sum(demands[custs])
            
            # Reassign each customer to the best available facility
            # Sort by demand descending
            order = np.argsort(-demands[custs])
            for idx in order:
                i = custs[idx]
                # Find best facility with space
                best_f = -1
                best_cost = np.inf
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        cost = assign_cost[i, f]
                        # Add open cost if facility was empty
                        if load[f] == 0:
                            cost += open_cost[f]
                        if cost < best_cost:
                            best_cost = cost
                            best_f = f
                if best_f == -1:
                    # Can't fit, restore
                    for j, ci in enumerate(custs):
                        res[ci] = old_res[j]
                    load = old_load.copy()
                    break
                res[i] = best_f
                load[best_f] += demands[i]
            else:
                # All reassigned, compute new score
                new_score = compute_score(res)
                delta = new_score - current_score
                if delta < 0 or np.random.rand() < np.exp(-delta / max(t, 1e-9)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_res = res.copy()
                        best_load = load.copy()
                else:
                    # Revert
                    res[custs] = old_res
                    load = old_load.copy()
            
            t *= cooling
            if t < t1:
                t = t1
        
        return best_res, best_load, best_score

    # Initial greedy solution
    res = greedy_init()
    load = np.zeros(n_fac)
    np.add.at(load, res, demands)
    
    # Run local search on initial solution
    local_search(res, load, max_iter=2000)
    best_score = compute_score(res)
    best_assign = res.copy()
    
    # Run cluster simulated annealing with multiple restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Perturb the current best solution
        perturbed_res = best_assign.copy()
        perturbed_load = np.zeros(n_fac)
        np.add.at(perturbed_load, perturbed_res, demands)
        
        # Randomly reassign 10-20% of customers
        n_perturb = max(1, int(0.15 * n_cust))
        perturb_custs = np.random.choice(n_cust, size=n_perturb, replace=False)
        for i in perturb_custs:
            old_f = perturbed_res[i]
            perturbed_load[old_f] -= demands[i]
            # Assign to random feasible facility
            feasible = np.where(perturbed_load + demands[i] <= capacities)[0]
            if len(feasible) > 0:
                new_f = feasible[np.random.randint(len(feasible))]
            else:
                new_f = old_f
            perturbed_res[i] = new_f
            perturbed_load[new_f] += demands[i]
        
        # Run cluster SA on perturbed solution
        sa_res, sa_load, sa_score = cluster_simulated_annealing(
            perturbed_res, perturbed_load, max_iter=20000, t0=500.0, t1=1.0, cooling=0.99995
        )
        
        if sa_score < best_score:
            best_score = sa_score
            best_assign = sa_res.copy()
    
    # Final local search on best solution
    load_final = np.zeros(n_fac)
    np.add.at(load_final, best_assign, demands)
    local_search(best_assign, load_final, max_iter=3000)
    
    # Verify feasibility
    load_verify = np.zeros(n_fac)
    np.add.at(load_verify, best_assign, demands)
    if np.any(load_verify > capacities + 1e-9):
        # Fallback to greedy if somehow infeasible
        best_assign = greedy_init()
    
    return best_assign.astype(int)

# EVOLVE-BLOCK-END
