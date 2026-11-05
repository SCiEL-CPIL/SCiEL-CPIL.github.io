<<<<<<< SEARCH
    # Final high-precision polish: use Nelder-Mead on a constrained parameterization
    # Use log-weights and cumulative nodes to enforce constraints
    if time.time() < deadline - 5.0:
        # Convert to constrained-free parameters
        # nodes: use cumulative sums of positive values, normalized to [0,1]
        # weights: use log parameters, softmax to ensure positive and sum=1
        
        def objective_constrained(params):
            # params: 15 values (7 node gaps + 8 log weights)
            node_gaps = np.exp(params[:7])  # positive gaps
            log_w = params[7:]
            
            # Normalize nodes to [0,1]
            cum_nodes = np.concatenate([[0.0], np.cumsum(node_gaps)])
            total = cum_nodes[-1]
            if total < 1e-10:
                return 100.0
            nodes_c = cum_nodes / total
            # Ensure strictly in (0,1) with small margin
            nodes_c = np.clip(nodes_c, 1e-6, 1.0 - 1e-6)
            # Re-normalize to ensure sum of gaps = 1 (they should by construction)
            
            # Softmax weights
            log_w_shifted = log_w - np.max(log_w)
            w_exp = np.exp(log_w_shifted)
            weights_c = w_exp / np.sum(w_exp)
            
            x = np.concatenate([nodes_c, weights_c])
            return objective(x)
        
        # Initialize from best solution
        node_gaps_init = np.diff(best_x[:8])
        node_gaps_init = np.maximum(node_gaps_init, 1e-4)
        log_w_init = np.log(np.maximum(best_x[8:], 1e-10))
        x_con_init = np.concatenate([node_gaps_init, log_w_init])
        
        try:
            res2 = minimize(objective_constrained, x_con_init, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res2.fun < best_val:
                # Convert back
                node_gaps = np.exp(res2.x[:7])
                log_w = res2.x[7:]
                cum_nodes = np.concatenate([[0.0], np.cumsum(node_gaps)])
                nodes_final = cum_nodes / cum_nodes[-1]
                nodes_final = np.clip(nodes_final, 1e-6, 1.0 - 1e-6)
                log_w_shifted = log_w - np.max(log_w)
                w_exp = np.exp(log_w_shifted)
                weights_final = w_exp / np.sum(w_exp)
                best_x = np.concatenate([nodes_final, weights_final])
                best_val = res2.fun
        except Exception:
            pass
=======
    # Final high-precision polish: use Nelder-Mead on a constrained parameterization
    # Use log-weights and cumulative nodes to enforce constraints
    if time.time() < deadline - 5.0:
        # Convert to constrained-free parameters
        # nodes: use cumulative sums of positive values, normalized to [0,1]
        # weights: use log parameters, softmax to ensure positive and sum=1
        
        def objective_constrained(params):
            # params: 15 values (7 node gaps + 8 log weights)
            node_gaps = np.exp(params[:7])  # positive gaps
            log_w = params[7:]
            
            # Normalize nodes to [0,1]
            cum_nodes = np.concatenate([[0.0], np.cumsum(node_gaps)])
            total = cum_nodes[-1]
            if total < 1e-10:
                return 100.0
            nodes_c = cum_nodes / total
            # Ensure strictly in (0,1) with small margin
            nodes_c = np.clip(nodes_c, 1e-6, 1.0 - 1e-6)
            
            # Softmax weights
            log_w_shifted = log_w - np.max(log_w)
            w_exp = np.exp(log_w_shifted)
            weights_c = w_exp / np.sum(w_exp)
            
            x = np.concatenate([nodes_c, weights_c])
            return objective(x)
        
        # Initialize from best solution
        node_gaps_init = np.diff(best_x[:8])
        node_gaps_init = np.maximum(node_gaps_init, 1e-4)
        log_w_init = np.log(np.maximum(best_x[8:], 1e-10))
        x_con_init = np.concatenate([node_gaps_init, log_w_init])
        
        try:
            res2 = minimize(objective_constrained, x_con_init, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
            if res2.fun < best_val:
                # Convert back
                node_gaps = np.exp(res2.x[:7])
                log_w = res2.x[7:]
                cum_nodes = np.concatenate([[0.0], np.cumsum(node_gaps)])
                nodes_final = cum_nodes / cum_nodes[-1]
                nodes_final = np.clip(nodes_final, 1e-6, 1.0 - 1e-6)
                log_w_shifted = log_w - np.max(log_w)
                w_exp = np.exp(log_w_shifted)
                weights_final = w_exp / np.sum(w_exp)
                best_x = np.concatenate([nodes_final, weights_final])
                best_val = res2.fun
        except Exception:
            pass
        
        # Additional polish: try COBYLA if time permits
        if time.time() < deadline - 3.0:
            def objective_cobyla(params):
                nodes_c = params[:8]
                weights_c = params[8:]
                # Simple penalty for out-of-bounds
                penalty = 0.0
                if np.any(nodes_c < 0.01) or np.any(nodes_c > 0.99):
                    penalty += 10.0
                if np.any(np.diff(nodes_c) <= 0):
                    penalty += 10.0
                if np.any(weights_c <= 0):
                    penalty += 10.0
                w_sum = np.sum(weights_c)
                penalty += 5.0 * abs(w_sum - 1.0)
                if penalty > 1.0:
                    return penalty
                x = np.concatenate([nodes_c, weights_c])
                return objective(x) + penalty
            
            try:
                res3 = minimize(objective_cobyla, best_x, method='COBYLA', 
                              options={'maxiter': 1000, 'rhobeg': 1e-6})
                if res3.fun < best_val:
                    best_x = res3.x.copy()
                    best_val = res3.fun
            except Exception:
                pass
>>>>>>> REPLACE

