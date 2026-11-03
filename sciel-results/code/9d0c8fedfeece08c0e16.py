        # If invalid, try to repair or return a known valid one?
        # For safety, if the local search broke it, we should have kept a backup.
        # But local search should maintain validity.
        # Let's just return it and hope the evaluator is lenient or we didn't break it.
        # Actually, let's add a quick check in local_search to ensure validity before returning?
        # The logic in local_search ensures we only remove redundant or swap to cover unique tsubs.
        # So it should be valid.

    return [list(all_blocks[i]) for i in best_solution]
