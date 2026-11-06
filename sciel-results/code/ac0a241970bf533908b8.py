import os
import sys

# Try to locate the baseline module to understand the dag structure and naive schedule
try:
    from autoresearch_suite.science_tcs_expansion import pebble_baseline as _baseline
except Exception:
    _baseline = None
