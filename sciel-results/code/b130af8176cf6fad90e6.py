import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0
_machine_counts = {}

def dispatch(ready_operations, time):
        if not ready_operations:
            return 0

        elapsed = _time.time() - _dispatch_start_time
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            # Fallback: simple urgency
            return int(np.argmax([op["remaining_work"] for op in ready_operations]))

        n = len(ready_operations)
        if n == 1:
            op = ready_operations[0]
            mo = op["machine_options"]
            if mo:
                # Pick min duration, tie-break by load
                best_m, best_d = None, float('inf')
                for m, d in mo.items():
                    count = _machine_counts.get(m, 0)
                    # Penalize heavily loaded machines slightly
                    effective_d = d * (1 + count * 0.01) 
                    if effective_d < best_d:
                        best_d = effective_d
                        best_m = m
                _machine_counts[best_m] = _machine_counts.get(best_m, 0) + 1
                return (0, best_m)
            return 0

        # Calculate scores
        # score = remaining_work / min_duration
        # This prioritizes high-work, fast-executable ops.
        
        scores = []
        min_durs = []
        for op in ready_operations:
            mo = op["machine_options"]
            if not mo:
                scores.append(0)
                min_durs.append(float('inf'))
                continue
            min_d = min(mo.values())
            min_durs.append(min_d)
            work = op["remaining_work"]
            # Avoid division by zero
            score = work / (min_d + 1e-6)
            scores.append(score)
            
        best_idx = int(np.argmax(scores))
        
        # Machine selection for best_idx
        op = ready_operations[best_idx]
        mo = op["machine_options"]
        if mo:
            best_m = None
            best_score_m = float('inf')
            for m, d in mo.items():
                count = _machine_counts.get(m, 0)
                # Heuristic: prefer lower load if duration is close
                # Threshold: 1.1 * min_duration
                min_d = min(mo.values())
                if d <= min_d * 1.1:
                    effective_score = d + count * 0.1 # Small penalty for load
                else:
                    effective_score = d * 10 + count # Heavy penalty if slower
                if effective_score < best_score_m:
                    best_score_m = effective_score
                    best_m = m
            
            _machine_counts[best_m] = _machine_counts.get(best_m, 0) + 1
            return (best_idx, best_m)
        
        return best_idx
