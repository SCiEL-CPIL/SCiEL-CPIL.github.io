            if bad_count < 50:
                # Scan for a bad TP
                found_bad = False
                for i in range(num_tps):
                    if tp_distinct[i] < 3:
                        candidate_tp_idx = i
                        found_bad = True
                        break
                if not found_bad:
                    # Should not happen if bad_count > 0, but safety
                    candidate_tp_idx = random.randint(0, num_tps - 1)
            else:
                # Random pick
                candidate_tp_idx = random.randint(0, num_tps - 1)
                # If we picked a good one, we might still try to change a pos in it?
                # No, let's just retry if it's good and bad_count is high?
                # Or just proceed with the change, hoping to improve?
                # If it's good, changing a pos might make it bad.
                # So we should only change if it helps or is neutral?
                # The SA logic handles this.
                # But if we always pick good TPs when bad_count is high, we waste time.
                # Let's just pick a random pos and check its TPs as before, but limit retries?
                # Actually, the previous logic of picking a random pos and checking its TPs is better for high bad_count.
                # Let's stick to: Pick random pos. If it has a bad TP, use it. Else, pick random TP.
                
                pos = random.randint(0, n - 1)
                candidate_tp_idx = -1
                for tp_idx in pos_to_tps[pos]:
                    if tp_distinct[tp_idx] < 3:
                        candidate_tp_idx = tp_idx
                        break
                if candidate_tp_idx == -1:
                    candidate_tp_idx = random.randint(0, num_tps - 1)
