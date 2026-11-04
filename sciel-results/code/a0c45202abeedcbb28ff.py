    # Phase 0: Phase-only spectral design
    if time.time() < deadline - 60:
        x_phase, val_phase = _optimize_phase_only(deadline=deadline - 30)
        if x_phase is not None and val_phase < best_val:
            best_val = val_phase
            best_x = x_phase.copy()
