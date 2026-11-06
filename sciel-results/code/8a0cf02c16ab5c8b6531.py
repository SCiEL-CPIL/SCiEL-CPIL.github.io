import inspect
try:
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    print(inspect.getsource(pebble_baseline))
except Exception as e:
    print("ERR", e)
