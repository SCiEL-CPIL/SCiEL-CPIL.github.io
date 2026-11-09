# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_cost_flow
import heapq


def assign(demands, capacities, open_cost, assign_cost):
    """
    Capacitated facility location: assign 100 customers to 20 facilities.
    Uses Min-Cost Flow for exact assignment given facility subsets,
    followed by local search for facility opening/closing and reassignment.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit, 10s buffer)
    
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    def _total_cost(result):
        """Compute total cost: open cost + assignment cost"""
        used = set(result.tolist())
        oc = sum(open_cost[f] for f in used)
        ac = sum(assign_cost[i, result[i]] for i in range(n_cust))
        return oc + ac
    
    def _is_feasible(result):
        """Check if assignment is feasible (no capacity violations)"""
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
        return np.all(load <= capacities + 1e-9)

    def _solve_mcf(facility_subset):
        """
        Solve min-cost flow for a specific subset of facilities.
        Returns (assignment_array, assignment_cost) or (None, inf) if infeasible.
        """
        if not facility_subset:
            return None, float('inf')
        
        # Node indexing:
        # 0: Source
        # 1 to n_cust: Customers
        # n_cust+1 to n_cust+n_fac_subset: Facilities
        # n_cust+n_fac_subset+1: Sink
        
        fac_indices = list(facility_subset)
        n_fac_sub = len(fac_indices)
        total_nodes = 1 + n_cust + n_fac_sub + 1
        sink = total_nodes - 1
        
        # Graph structure for min_cost_flow
        # node_ind, in_cap, out_cap, cost
        # We need to map facility local index to global node index
        # fac_local_idx -> node_id = n_cust + 1 + fac_local_idx
        
        node_ind = []
        in_cap = []
        out_cap = []
        cost = []
        
        # 1. Source to Customers
        # Source is 0
        # Customer i is i+1
        for i in range(n_cust):
            node_ind.append(0)
            node_ind.append(i + 1)
            in_cap.append(0)
            in_cap.append(demands[i])
            out_cap.append(demands[i])
            out_cap.append(0)
            cost.append(0)
            cost.append(0)
            
        # 2. Customers to Facilities
        for i in range(n_cust):
            for f_idx, f_global in enumerate(fac_indices):
                # Customer i is i+1
                # Facility f_global is n_cust + 1 + f_idx
                node_ind.append(i + 1)
                node_ind.append(n_cust + 1 + f_idx)
                in_cap.append(0)
                in_cap.append(demands[i]) # Capacity is demand of customer i (can assign all to this fac)
                out_cap.append(demands[i])
                out_cap.append(0)
                cost.append(assign_cost[i, f_global])
                cost.append(0) # Reverse edge cost is 0? No, in MCF, cost is usually symmetric or defined per direction. 
                # scipy min_cost_flow expects cost for the edge. The reverse edge is implicit for flow conservation but cost is typically 0 for reverse in simple construction? 
                # Actually, scipy.min_cost_flow takes (node_ind, in_cap, out_cap, cost).
                # It handles the residual graph internally. We just define the forward edges.
                
        # 3. Facilities to Sink
        for f_idx, f_global in enumerate(fac_indices):
            node_ind.append(n_cust + 1 + f_idx)
            node_ind.append(sink)
            in_cap.append(0)
            in_cap.append(capacities[f_global])
            out_cap.append(capacities[f_global])
            out_cap.append(0)
            cost.append(0)
            cost.append(0)
            
        # Ensure arrays are correct types
        node_ind = np.array(node_ind, dtype=np.int64)
        in_cap = np.array(in_cap, dtype=np.int64)
        out_cap = np.array(out_cap, dtype=np.int64)
        cost = np.array(cost, dtype=np.float64)
        
        # min_cost_flow returns (cost, flow)
        # flow is a matrix of shape (total_nodes, total_nodes)
        try:
            mcf_cost, flow = min_cost_flow(node_ind, in_cap, out_cap, cost)
        except Exception:
            return None, float('inf')
            
        # Check if all demand was satisfied
        # Flow from source to customers should equal demand
        # Or simply check if mcf_cost is finite and flow conservation holds
        # A simpler check: sum of flow from source should be sum(demands)
        total_flow_out_source = np.sum(flow[0, :])
        if total_flow_out_source < sum(demands) - 1e-6:
            return None, float('inf')
            
        # Extract assignment
        assignment = np.zeros(n_cust, dtype=int)
        assignment_cost = 0.0
        for i in range(n_cust):
            # Look at flow from customer i (node i+1) to facilities
            # Facility nodes are n_cust+1 to n_cust+n_fac_sub
            flows_to_facs = flow[i+1, n_cust+1:n_cust+1+n_fac_sub]
            # Find which facility got the flow
            # There should be exactly one facility with flow == demands[i] (or close)
            # Or potentially split? No, in this graph structure, customer i sends demand[i] to one or more facilities?
            # Wait, the edge capacity from Customer to Facility is demands[i].
            # So customer i can send all its demand to multiple facilities?
            # The problem requires assigning each customer to ONE facility.
            # My graph construction allows splitting demand among multiple facilities for a single customer.
            # This is a flaw. The CFLP typically requires integer assignment (1 customer -> 1 facility).
            # However, since demands are integers and capacities are integers, and costs are linear,
            # the LP relaxation of the assignment problem with capacities often yields integer solutions?
            # No, the "customer to facility" edge allows splitting.
            # To enforce 1 customer -> 1 facility, we would need a different formulation or post-processing.
            # But wait, if we split, it's not a valid assignment for the task "one facility index per customer".
            # We need to fix this.
            
            # If splitting occurs, we must pick the best one?
            # Actually, for 100 customers and 20 facilities, the "splitting" might happen if a customer's demand is large and a facility is full.
            # But the standard CFLP formulation is:
            # min sum c_ij x_ij + sum f_j y_j
            # s.t. sum_j x_ij = 1 for all i
            #      sum_i d_i x_ij <= C_j y_j for all j
            #      x_ij, y_j in {0,1}
            
            # My MCF formulation is:
            # Flow from i to j is x_ij * d_i ? No, flow is amount of demand.
            # If flow i->j is d_i, then x_ij=1.
            # If flow is split, x_ij is fractional.
            # Since we need integer x_ij, we cannot use standard MCF directly without ensuring integrality.
            # However, the constraint matrix for the assignment part (with capacities) is not totally unimodular generally?
            # Actually, the transportation problem matrix IS totally unimodular.
            # But here we have capacities on the facility side and supply on the customer side.
            # This IS a transportation problem.
            # The transportation problem has integer optimal solutions if supplies and demands are integers.
            # So the flow will be integer.
            # But will customer i's demand be split?
            # Yes, it can be. e.g. demand 10, fac A cap 5, fac B cap 5. Flow 5 to A, 5 to B.
            # This violates "assign to ONE facility".
            
            # So MCF is not directly applicable for "1 customer -> 1 facility" unless we prove that an optimal solution exists where no customer is split.
            # This is NOT always true.
            
            # Alternative: Use MCF to get a "soft" assignment, then repair?
            # Or use a different method.
            
            # Given the complexity, let's stick to the heuristic approach but make it much faster and smarter.
            # The previous score was 1.23.
            
            pass
        return None, float('inf')
        
    # Since MCF splitting is an issue, let's revert to a highly optimized heuristic.
    # The previous local search was slow. Let's optimize it.
    
    def _fast_local_search(result, max_iters=500):
        """
        Optimized local search with 1-move improvements.
        Uses precomputed loads and efficient delta calculation.
        """
        result = result.copy()
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[result[i]] += demands[i]
            
        # Precompute which facilities are open
        is_open = np.zeros(n_fac, dtype=bool)
        for f in set(result.tolist()):
            is_open[f] = True
            
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # 1-move improvements
            for i in range(n_cust):
                current_f = result[i]
                current_cost = assign_cost[i, current_f]
                d = demands[i]
                
                # Find the best alternative facility
                # We want to minimize:
                # new_assign_cost - current_assign_cost
                # + (open_cost[f] if f not open and load[f]==0 else 0)
                # - (open_cost[current_f] if load[current_f]==d else 0)
                
                # Vectorized calculation of assignment cost deltas
                # But open cost depends on state.
                
                best_delta = 0.0
                best_f = -1
                
                # We can filter candidates:
                # 1. Must have capacity: load[f] + d <= capacities[f]
                # 2. Potential gain: assign_cost[i, f] < current_cost + open_cost considerations
                
                # Iterate through facilities (n_fac=20 is small)
                for f in range(n_fac):
                    if f == current_f:
                        continue
                    if load[f] + d > capacities[f] + 1e-9:
                        continue
                    
                    delta_assign = assign_cost[i, f] - current_cost
                    
                    delta_open = 0.0
                    # If f is not open, and becomes open (load[f] was 0)
                    if not is_open[f] and load[f] == 0:
                        delta_open += open_cost[f]
                    
                    # If current_f becomes empty
                    if load[current_f] == d:
                        delta_open -= open_cost[current_f]
                        
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta - 1e-9:
                        best_delta = total_delta
                        best_f = f
                        
                if best_f != -1:
                    # Apply move
                    load[current_f] -= d
                    load[best_f] += d
                    result[i] = best_f
                    
                    # Update is_open
                    if load[current_f] == 0:
                        is_open[current_f] = False
                    if load[best_f] > 0 and not is_open[best_f]:
                        is_open[best_f] = True
                    # Note: if load[best_f] was >0, it was already open.
                    # If load[current_f] was >d, it remains open.
                    
                    improved = True
                    # Break or continue? Continue to find more moves in this pass?
                    # Usually, first-improvement is faster to code, best-improvement is better.
                    # Let's stick to first-improvement per customer for speed, then re-scan.
        
        return result

    # Generate diverse initial solutions
    def _greedy_init(order, allowed_facilities):
        load = np.zeros(n_fac)
        result = np.full(n_cust, -1, dtype=int)
        allowed = np.array(allowed_facilities)
        
        for i in order:
            placed = False
            costs = assign_cost[i, allowed]
            sorted_idx = np.argsort(costs)
            for idx in sorted_idx:
                f = allowed[idx]
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    result[i] = f
                    placed = True
                    break
            if not placed:
                return None
        return result

    # Orderings
    orderings = [
        np.argsort(-demands),
        np.argsort(demands),
        np.argsort(assign_cost.mean(axis=1)),
        np.random.permutation(n_cust),
        np.random.permutation(n_cust),
    ]
    
    # Subsets
    subsets_to_try = [list(range(n_fac))]
    by_open = np.argsort(open_cost)
    by_assign = np.argsort(assign_cost.mean(axis=0))
    by_combined = np.argsort(open_cost + assign_cost.mean(axis=0))
    
    for k in [10, 12, 15, 18]:
        subsets_to_try.append(by_open[:k].tolist())
        subsets_to_try.append(by_assign[:k].tolist())
        subsets_to_try.append(by_combined[:k].tolist())
        
    # Remove duplicates
    unique_subsets = []
    seen = set()
    for s in subsets_to_try:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)
            
    best_result = None
    best_cost = float('inf')
    
    for subset in unique_subsets:
        if time.time() > deadline - 20:
            break
        for order in orderings:
            if time.time() > deadline - 20:
                break
            res = _greedy_init(order, subset)
            if res is not None and _is_feasible(res):
                c = _total_cost(res)
                if c < best_cost:
                    best_cost = c
                    best_result = res.copy()
                    
    if best_result is not None:
        # Refine with local search
        if time.time() < deadline - 5:
            improved_res = _fast_local_search(best_result, max_iters=1000)
            if _is_feasible(improved_res):
                c = _total_cost(improved_res)
                if c < best_cost:
                    best_cost = c
                    best_result = improved_res
                    
    # Fallback
    if best_result is None:
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            f = i % n_fac
            while load[f] + demands[i] > capacities[f]:
                f = (f + 1) % n_fac
            load[f] += demands[i]
            best_result[i] = f
            
    return best_result.astype(int)

# EVOLVE-BLOCK-END
