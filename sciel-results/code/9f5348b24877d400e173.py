I should probably keep this or replace it with a quick L-BFGS.
Let's replace the Phase 3 in the main loop with a quick L-BFGS as well, or just keep `_polish` if it's fast enough.
Actually, `_polish` is quite slow because it evaluates energy many times. L-BFGS is generally better.
Let's replace the Phase 3 call with a short L-BFGS run.

